import argparse
import torch
# from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm 
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
import numpy as np
from comet import download_model, load_from_checkpoint

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tokenizer', required=False, default="haoranxu/ALMA-7B")
    parser.add_argument('--src', required=True)
    parser.add_argument('--data_path', type=str, default="hf://datasets/haoranxu/WMT22-Test/cs-en/test-00000-of-00001-1a83a591805d9178.parquet")
    parser.add_argument('--tgt', required=True)
    parser.add_argument('--dtype', required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--beam', type=int, required=True)
    parser.add_argument('--gen_max_tokens', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--eval_samples', type=int, required=False)
    return parser

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

def main():
    parser = get_parser()
    args = parser.parse_args()

    initial_vram = torch.cuda.memory_allocated()
    # set data dtype
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map.get(args.dtype, torch.float)

    model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                 torch_dtype=dtype, 
                                                 device_map="auto",
                                                 offload_folder="./offload"
                                                 )
    
    size_all_mb = print_model_size(model)
    # if "state_dict" in checkpoint:
    #     model_scorer.load_state_dict(checkpoint["state_dict"])
    # else:
    #     # Handle the case where the checkpoint is not a state_dict (model is fully saved)
    #     model = checkpoint


    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    initial_vram = torch.cuda.memory_allocated()

    print(download_model.available_legacy_metrics())
    model_scorer = download_model("Unbabel/wmt22-comet-da") # torch.load("model.ckpt", weights_only=True)
    model_scorer = load_from_checkpoint(model_scorer)

    src = LANG_MAP[args.src]
    tgt = LANG_MAP[args.tgt]

    # read data
    ds = pd.read_parquet(path=args.data_path)

    # Initialize an empty list for the lines
    lines = []
    targets = []
    path = args.src + "-" + args.tgt
    len_samples = len(ds[path]) if not args.eval_samples else args.eval_samples


    # Iterate over the dataset and extract the relevant information
    for idx, example in enumerate(ds[path][:len_samples]):
        czech_sentence = example[args.src]  # Source sentence in Czech
        english_translation = example[args.tgt]  # Target sentence in English

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
    total_vram_per_batch = []
    # vram_per_batch = []
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

        torch.cuda.synchronize()
        with torch.no_grad():
            # before_infernce_vram = torch.cuda.memory_allocated()
            start.record()
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

    comet = model_scorer.predict(comet_score, batch_size=8, gpus=1)
    print (f"Comet score:{comet}")

    print("*"*100)

if __name__ == "__main__":
    main()