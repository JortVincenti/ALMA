import argparse
import os
import torch
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer

from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig 
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm 

import numpy as np
# from smoothquant.smooth import smooth_lm
# from smoothquant.fake_quant import quantize_llama_like
from comet import download_model, load_from_checkpoint


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tokenizer', required=False, default="ALMA-7B")
    parser.add_argument('--source_lang', required=True)
    parser.add_argument('--target_lang', required=True)
    parser.add_argument('--dtype', required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--beam', type=int, required=True)
    parser.add_argument('--gen_max_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--eval_samples', type=int, required=False)
    parser.add_argument('--quant_type', type=str, default=None, help='Quantization type. Options are None, "naive","fake_smooth", or "smooth_8bit". Defaults to None.')
    parser.add_argument(
        "--mode",
        type=str,
        default="mode_2",
        help="Controls the model type for act scales. Options: mode_1, mode_2, mode_3, mode_4, mode_5",
    )
    parser.add_argument("--act-scales", type=str,
                        default='full_scale_activation_models/ALMA-7B/')
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
    print('Model size: {:.3f}MB'.format(size_all_mb))

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


    if args.quant_type =="LLM_int8":
        quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                              bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=getattr(torch, "float16"),
                                              bnb_4bit_use_double_quant=True)
        
        model = AutoModelForCausalLM.from_pretrained(args.model, 
                                            torch_dtype=dtype, 
                                            device_map="auto",
                                            offload_folder="./offload",
                                            quantization_config=quant_config
                                            )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                    torch_dtype=dtype, 
                                                    device_map="auto",
                                                    offload_folder="./offload"
                                                    )
    
    if args.quant_type is not None and not args.quant_type=="LLM_int8":
        print(f"Running with quantization type: {args.quant_type}")
        if args.quant_type == 'fake_smooth':
            act_scales = torch.load(os.path.join(args.act_scales, f"{args.mode}.pth"))
            smooth_lm(model, act_scales, 0.85)
            model = quantize_llama_like(model)
        elif args.quant_type == 'naive':
            model = quantize_llama_like(model)
        elif args.quant_type == 'smooth_8bit':
            act_scales = torch.load(os.path.join(args.act_scales, f"{args.mode}.pth"))
            smooth_lm(model, act_scales, 0.85)
            
            user = os.getenv('USER')
            scratch_path = os.path.join(f"/scratch-shared/{user}/", args.act_scales)
            os.makedirs(scratch_path, exist_ok=True)
            smoothed_path = os.path.join(scratch_path, f"{args.source_lang}-{args.target_lang}-{args.mode}-smooth_8bit")
            print(f"Saving smoothed model to {smoothed_path}")
            model.save_pretrained(smoothed_path, from_pt=True)
            print("Emptying cache")
            del model
            torch.cuda.empty_cache()
            print("Configuring quantization")
            quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                              bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=getattr(torch, "float16"),
                                              bnb_4bit_use_double_quant=True)
            print(f"Loading smoothed model from {smoothed_path}")
            model = AutoModelForCausalLM.from_pretrained(smoothed_path, 
                                                torch_dtype=dtype, 
                                                device_map="auto",
                                                offload_folder="./offload",
                                                quantization_config=quant_config
                                                )
            
        else:
            raise ValueError(f"Invalid quantization type: {args.quant_type}")
    else:
        print(f"Running with quantization type: {str(args.quant_type) if args.quant_type is None else args.quant_type}")
    
    print_model_size(model)

    model.eval()
    print("model:", args.model)
    # tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer_path = "haoranxu/ALMA-7B"
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    source_lang = LANG_MAP[args.source_lang]
    target_lang = LANG_MAP[args.target_lang]

    #file_out = open(args.fout, "w")

    # read data
    if (args.source_lang=="cs" and args.target_lang=="en"):
        dataset_path = "hf://datasets/haoranxu/WMT22-Test/cs-en/test-00000-of-00001-1a83a591805d9178.parquet"
    if (args.source_lang=="en" and args.target_lang=="cs"):
        dataset_path = "hf://datasets/haoranxu/WMT22-Test/en-cs/test-00000-of-00001-b92f389a2a10e4b5.parquet"
    if (args.source_lang=="de" and args.target_lang=="en"):
        print("came in here")
        dataset_path = "hf://datasets/haoranxu/WMT22-Test/de-en/test-00000-of-00001-c03dcec47c23d6ca.parquet"
    if (args.source_lang=="en" and args.target_lang=="de"):
        dataset_path = "hf://datasets/haoranxu/WMT22-Test/en-de/test-00000-of-00001-c470e1e53ed73302.parquet"
    if (args.source_lang=="en" and args.target_lang=="ru"):
        dataset_path = "hf://datasets/haoranxu/WMT22-Test/en-ru/test-00000-of-00001-889b8af39e8c83c4.parquet"
    if (args.source_lang=="ru" and args.target_lang=="en"):
        dataset_path = "hf://datasets/haoranxu/WMT22-Test/ru-en/test-00000-of-00001-4455a1b04d42177e.parquet"
    if (args.source_lang=="en" and args.target_lang=="zh"):
        dataset_path = "hf://datasets/haoranxu/WMT22-Test/en-zh/test-00000-of-00001-6b3b7f42ead58b33.parquet"
    if (args.source_lang=="zh" and args.target_lang=="en"):
        dataset_path = "hf://datasets/haoranxu/WMT22-Test/zh-en/test-00000-of-00001-a8c846c3e121c2f6.parquet"
    if (args.source_lang=="en" and args.target_lang=="is"):
        dataset_path = "hf://datasets/haoranxu/WMT22-Test/en-is/test-00000-of-00001-872ab78ba9548351.parquet"
    if (args.source_lang=="is" and args.target_lang=="en"):
        dataset_path = "hf://datasets/haoranxu/WMT22-Test/is-en/test-00000-of-00001-bb3b8280f4b7ff31.parquet"
    
    ds = pd.read_parquet(dataset_path)
    
    # Initialize an empty list for the lines
    lines = []
    targets = []
    path = args.source_lang + "-" + args.target_lang
    len_samples = len(ds[path]) if not args.eval_samples else args.eval_samples


    # Iterate over the dataset and extract the relevant information
    for example in ds[path][:len_samples]:
        czech_sentence = example[args.source_lang]  # Source sentence in Czech
        english_translation = example[args.target_lang]  # Target sentence in English
        
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
    latency_per_batch = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    increment = 0
    comet_score = []
    for batch in tqdm(dynamic_batching(tokenizer, lines, args.batch_size, args.gen_max_tokens), total=total_batches, desc="Processing Batches"):
        prompts = []
        for line in batch:
            line = line.strip()
            # prepend prompt
            prompt = f"Translate this from {source_lang} to {target_lang}:\n{source_lang}: {line}\n{target_lang}:"
            prompts.append(prompt)

        # Tokenize with truncation and dynamic padding up to the longest sequence in the batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, ).to('cuda') if torch.cuda.is_available() else tokenizer(prompts, return_tensors="pt", padding=True, ).to('cpu')

        torch.cuda.synchronize()
        with torch.no_grad():
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

            comet_score.append(
                {"source_lang": prompt,
                 "mt": translation,
                 'ref' : targets[increment]
                 }
            )
            increment += 1

    print("-----------------------------------------------------------------")
    print_model_size(model)
    
    model_average_vram = sum(total_vram_per_batch) / len(total_vram_per_batch)
    print(f"Average VRAM usage: {model_average_vram} MB with model")

    print("-----------------------------------------------------------------")
    
    temp_times = np.array(latency_per_batch)
    mean_time = temp_times[abs(temp_times - np.mean(temp_times)) < np.std(temp_times)].mean()

    del model
    torch.cuda.empty_cache()
    

    print("*"*100)
    print("Evaluation Results:")
    print(f"Avg Time taken for generation: {mean_time:.2f} ms")

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
    
    
    model_path = download_model("Unbabel/wmt22-comet-da")
    model_scorer = load_from_checkpoint(model_path)

    comet = model_scorer.predict(comet_score, batch_size=args.batch_size, gpus=1)
    print (f"Comet score:{comet}")
    average_comet_score = sum(comet['scores']) / len(comet['scores'])
    print(f"Average COMET score: {average_comet_score}")
    

    print("*"*100)


if __name__ == "__main__":
    main()