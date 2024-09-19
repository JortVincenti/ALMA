import deepspeed
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm
import time
import sacrebleu
from rouge_score import rouge_scorer
import pandas as pd
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--tgt', required=True)
    parser.add_argument('--dtype', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--beam', type=int, required=True)
    parser.add_argument('--gen_max_tokens', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--eval_samples', type=int, required=False)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # set data dtype
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map.get(args.dtype, torch.float)

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Init DeepSpeed
    model = deepspeed.init_inference(model, dtype=dtype, mp_size=1, replace_method='auto')

    src = args.src
    tgt = args.tgt

    # read data
    ds = pd.read_parquet("hf://datasets/haoranxu/WMT22-Test/cs-en/test-00000-of-00001-1a83a591805d9178.parquet")

    lines = []
    targets = []
    len_samples = len(ds["cs-en"]) if not args.eval_samples else args.eval_samples

    # Prepare input text and targets
    for idx, example in enumerate(ds["cs-en"][:len_samples]):
        czech_sentence = example['cs']  # Source sentence in Czech
        english_translation = example['en']  # Target sentence in English
        
        # Store input and target
        lines.append(f"{czech_sentence}\n")
        targets.append(f"{english_translation}\n")

    total_batches = (len(lines) + args.batch_size - 1) // args.batch_size
    generated_translations = []
    total_time = 0

    # Generate translations
    for batch in tqdm(dynamic_batching(tokenizer, lines, args.batch_size, args.gen_max_tokens), total=total_batches, desc="Processing Batches"):
        prompts = [f"Translate this from {src} to {tgt}:\n{src}: {line.strip()}\n{tgt}:" for line in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')

        with torch.no_grad():
            start = time.time()
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=args.beam,
                max_new_tokens=args.gen_max_tokens
            )
            total_time += time.time() - start

        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_translations.extend([output[len(prompt):].strip() for prompt, output in zip(prompts, outputs)])

    # Print timing and evaluation results
    print(f"Total time for generation: {total_time:.2f} seconds")
    print(f"Average time per generation: {total_time / len(lines):.2f} seconds")

    # Evaluate BLEU and ROUGE
    bleu = sacrebleu.corpus_bleu(generated_translations, [targets])
    print(f"BLEU score: {bleu.score}")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(gt, ref) for gt, ref in zip(generated_translations, targets)]
    
    avg_rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    avg_rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    avg_rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)

    print(f"Average ROUGE-1 F1 score: {avg_rouge1}")
    print(f"Average ROUGE-2 F1 score: {avg_rouge2}")
    print(f"Average ROUGE-L F1 score: {avg_rougeL}")


if __name__ == "__main__":
    main()
