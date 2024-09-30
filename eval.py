import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm 
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
import time

def get_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--fin', required=True)
    #parser.add_argument('--fout', required=True)
    #parser.add_argument('--ckpt', required=True)
    parser.add_argument('--src', required=True)
    parser.add_argument('--tgt', required=True)
    parser.add_argument('--dtype', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--beam', type=int, required=True)
    parser.add_argument('--gen_max_tokens', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--eval_samples', type=int, required=False)
    return parser

LANG_MAP = {
    'eng_Latn': 'English',
    'deu_Latn': 'German',
    'ces_Latn': 'Czech',
    'rus_Cyrl': 'Russian',
    'zho_Hans': 'Chinese'
}

def dynamic_batching(tokenizer, texts, batch_size, max_length):
    """
    dynamic padding up to the longest sequence in the batch.
    """
    batch = []
    batch_length = 0

    for text in texts:
        input_length = len(tokenizer.encode(text, truncation=True, max_length=max_length))
        if len(batch) > 0 and (batch_length + input_length > max_length or len(batch) == batch_size):
            yield batch
            batch = []
            batch_length = 0
        
        batch.append(text)
        batch_length = max(batch_length, input_length)

    if len(batch) > 0:
        yield batch

def main():
    parser = get_parser()
    args = parser.parse_args()

    # set data dtype
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map.get(args.dtype, torch.float)

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                 torch_dtype=dtype, 
                                                 device_map="auto",
                                                 offload_folder="./offload"
                                                 )
    #model = PeftModel.from_pretrained(model, args.ckpt) # load when you have lora
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    src = LANG_MAP[args.src]
    tgt = LANG_MAP[args.tgt]

    #file_out = open(args.fout, "w")

    # read data
    ds = pd.read_parquet("hf://datasets/haoranxu/WMT22-Test/cs-en/test-00000-of-00001-1a83a591805d9178.parquet")
    #ds = load_dataset("haoranxu/WMT22-Test", "cs-en")
    # Initialize an empty list for the lines
    lines = []
    targets = []

    len_samples = len(ds["cs-en"]) if not args.eval_samples else args.eval_samples


    # Iterate over the dataset and extract the relevant information
    for idx, example in enumerate(ds["cs-en"][:len_samples]):
        czech_sentence = example['cs']  # Source sentence in Czech
        english_translation = example['en']  # Target sentence in English
        
        # Format the line as needed, for example, showing both source and target
        line = f"{czech_sentence}\n"
        target = f"{english_translation}\n"
        # Append the formatted line to the lines list
        lines.append(line)
        targets.append(target)

    # generate
    total_batches = (len(lines) + args.batch_size - 1) // args.batch_size  # calculate the number of batches
    # Initialize empty lists to store the generated translations and targets
    generated_translations = []
    total_time = 0
    for batch in tqdm(dynamic_batching(tokenizer, lines, args.batch_size, args.gen_max_tokens), total=total_batches, desc="Processing Batches"):
        prompts = []
        for line in batch:
            line = line.strip()
            # prepend prompt
            prompt = f"Translate this from {src} to {tgt}:\n{src}: {line}\n{tgt}:"
            prompts.append(prompt)

        # Tokenize with truncation and dynamic padding up to the longest sequence in the batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, ).to('cuda') if torch.cuda.is_available() else tokenizer(prompts, return_tensors="pt", padding=True, ).to('cpu')

        # generate
        with torch.no_grad():
            start = time.time()
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=args.beam, # beam size
                max_new_tokens=args.gen_max_tokens
            )
            end_time = time.time() - start

        total_time += end_time        
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Process and write the translations
        for prompt, output in zip(prompts, outputs):
            translation = output[len(prompt):].strip()
            generated_translations.append(translation)

    print("*"*100)
    print("Evaluation Results:")
    print(f"Time taken for generation: {total_time:.2f} seconds")
    print(f"Average generation time: {total_time / len(lines)} seconds")

    # BLEU Score
    bleu = sacrebleu.corpus_bleu(generated_translations, [targets])
    print(f"BLEU score: {bleu.score}")

    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    for generated_translation, target in zip(generated_translations, targets):
        rouge_scores.append(scorer.score(generated_translation, target))

    # Display average ROUGE scores
    average_rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    average_rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    average_rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)

    print(f"Average ROUGE-1 F1 score: {average_rouge1}")
    print(f"Average ROUGE-2 F1 score: {average_rouge2}")
    print(f"Average ROUGE-L F1 score: {average_rougeL}")

    print("*"*100)


if __name__ == "__main__":
    main()