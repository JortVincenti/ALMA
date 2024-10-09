# import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm 
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
import time
import numpy as np

# def get_parser():
#     parser = argparse.ArgumentParser()
#     #parser.add_argument('--fin', required=True)
#     #parser.add_argument('--fout', required=True)
#     #parser.add_argument('--ckpt', required=True)
#     parser.add_argument('--src', required=True)
#     parser.add_argument('--data_path', type=str, default="hf://datasets/haoranxu/WMT22-Test/cs-en/test-00000-of-00001-1a83a591805d9178.parquet")
#     parser.add_argument('--tgt', required=True)
#     parser.add_argument('--dtype', required=True)
#     parser.add_argument('--model', type=str, required=True)
#     parser.add_argument('--beam', type=int, required=True)
#     parser.add_argument('--gen_max_tokens', type=int, default=256)
#     parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
#     parser.add_argument('--eval_samples', type=int, required=False)
#     parser.add_argument('--tokenizer', type=str, required=False, default='haoranxu/ALMA-7B')
#     return parser

LANG_MAP = {
    'en': 'English',
    'de': 'German',
    'cs': 'Czech',
    'ru': 'Russian',
    'zh': 'Chinese',
    'is': 'Icelandic'
}

def print_model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


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


def alma_eval(model, tokenizer, args, device=torch.device("cuda:0")):
    
    # set data dtype
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map.get(args.dtype, torch.float)

    initial_vram = torch.cuda.memory_allocated()
    size_all_mb = print_model_size(model)

    # NOT NECESSARY IF CALLED BY MAIN
    # load model and tokenizer
    # model = AutoModelForCausalLM.from_pretrained(args.model, 
    #                                              torch_dtype=dtype, 
    #                                              device_map="auto",
    #                                              offload_folder="./offload"
    #                                              )
    #model = PeftModel.from_pretrained(model, args.ckpt) # load when you have lora
    
    model.eval()
    # tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    src = LANG_MAP[args.source_lang]
    tgt = LANG_MAP[args.target_lang]

    #file_out = open(args.fout, "w")

    # read data
    # ds = pd.read_parquet(path=args.data_path)
    dataset_path = f"hf://datasets/haoranxu/WMT22-Test/{args.source_lang}-{args.target_lang}/test-00000-of-00001-1a83a591805d9178.parquet"
    test_df = pd.read_parquet(dataset_path)    # Initialize an empty list for the lines
    
    lines = []
    targets = []
    path = args.source_lang + "-" + args.target_lang
    len_samples = len(test_df[path]) if not args.eval_samples else args.eval_samples
    print("length of dataframes 'path' lang direction:", len(test_df[args.source_lang + "-" + args.target_lang]))
    print("length of test_df:", len(test_df))
    len_samples = len(test_df) if not args.eval_samples else args.eval_samples
    print(test_df.columns)

    # Iterate over the dataset and extract the relevant information
    
    for idx, example in enumerate(test_df[path][:len_samples]):
        source_sentence = example[args.source_lang]
        target_translation = example[args.target_lang]
        lines.append(f"{source_sentence}\n")
        targets.append(f"{target_translation}\n")

    # generate
    total_batches = (len(lines) + args.batch_size - 1) // args.batch_size  # calculate the number of batches
    # Initialize empty lists to store the generated translations and targets
    generated_translations = []
    total_vram_per_batch = []
    latency_per_batch = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    comet_score = []
    increment = 0

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
        torch.cuda.synchronize()
        with torch.no_grad():
            start.record()
            print("gen_max_tokens", args.gen_max_tokens)
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=args.beam, # beam size
                max_new_tokens=args.gen_max_tokens
            )
            end.record()
        torch.cuda.synchronize()
        
        latency_per_batch.append(start.elapsed_time(end))    
        final_vram = torch.cuda.memory_allocated()

        model_memory_per_batch = ((final_vram - initial_vram) // (1024 ** 2))
        total_vram_per_batch.append(model_memory_per_batch)

        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Process and write the translations
        for prompt, output in zip(prompts, outputs):
            translation = output[len(prompt):].strip()
            generated_translations.append(translation)

            comet_score.append({
                    "src": prompt,
                    "mt": translation,
                    "ref": targets[increment]
                } 
            )
            increment += 1

    model_average_vram = sum(total_vram_per_batch) / len(total_vram_per_batch)
    
    temp_times = np.array(latency_per_batch)
    mean_time = temp_times[abs(temp_times - np.mean(temp_times)) < np.std(temp_times)].mean()

    del model
    torch.cuda.empty_cache()

    print("*"*100)
    print("Evaluation Results:")
    print(f"Avg Time taken for generation: {mean_time:.2f} ms")
    print(f"Average VRAM usage: {model_average_vram} MB with model")
    print(f"Model size is: {size_all_mb:.2f}")

    # BLEU Score
    bleu = sacrebleu.corpus_bleu(generated_translations, [targets])
    print(f"BLEU score: {bleu.score}")

    model_path = download_model("Unbabel/wmt22-comet-da")
    model_scorer = load_from_checkpoint(model_path)
    
    comet = model_scorer.predict(comet_score, batch_size=args.batch_size, gpus=1)
    print (f"Comet score:{comet}")
    average_comet_score = sum(comet['scores']) / len(comet['scores'])
    print(f"Average COMET score: {average_comet_score}")

    print("*"*100)

    return bleu.score  # You can return more results as needed
