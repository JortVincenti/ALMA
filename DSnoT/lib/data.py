# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import datasets
from datasets import load_dataset
import pandas as pd

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_alma(nsamples, seed, seqlen, tokenizer, source_lang, target_lang):
    from datasets import Dataset
    from datasets import get_dataset_config_names

    import pandas as pd

    # Load the ALMA-Human-Parallel dataset with the specified language pair
    # dataset = load_dataset("haoranxu/ALMA-Human-Parallel", "de-en")

    configs = get_dataset_config_names("haoranxu/ALMA-Human-Parallel")
    print("Available configs: ", configs)

    print(f"Loading dataset with source_lang={source_lang} and target_lang={target_lang}")
    # Load the dataset as a Pandas DataFrame
    splits = {'train': 'cs-en/train-00000-of-00001-3a60b130a713425b.parquet', 'validation': 'cs-en/validation-00000-of-00001-d1f9a3fc339fbc84.parquet'}
    train_df = pd.read_parquet("hf://datasets/haoranxu/ALMA-Human-Parallel/" + splits["train"])
    val_df = pd.read_parquet("hf://datasets/haoranxu/ALMA-Human-Parallel/" + splits["validation"])

    # Convert the Pandas DataFrame back into a Hugging Face Dataset
    traindata = Dataset.from_pandas(train_df)
    valdata = Dataset.from_pandas(val_df)

    # Check if source_lang and target_lang exist in the dataset's fields
    sample_entry = traindata[0]
    print("Loaded dataset. Sample entry", sample_entry)
    available_fields = list(sample_entry.keys())
    if 'translation' not in available_fields:
        raise ValueError(f"Expected field 'translation', but found {available_fields}")

    # if source_lang not in available_fields or target_lang not in available_fields:
    #     raise ValueError(f"Expected fields {source_lang} and {target_lang}, but found {available_fields}")

    # Set seed for reproducibility
    random.seed(seed)

    # Prepare the data loader
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            translations = traindata[i]['translation']
            source_text = translations.get(source_lang)
            target_text = translations.get(target_lang)
            # source_text = traindata[i][source_lang]
            # target_text = traindata[i][target_lang]

            if source_text is None or target_text is None:
                raise ValueError(f"Source or target language not found in translation. Entry: {translations}")
            
            # Tokenize source and target text
            source_enc = tokenizer(source_text, return_tensors='pt', max_length=seqlen, truncation=True)
            target_enc = tokenizer(target_text, return_tensors='pt', max_length=seqlen, truncation=True)
            
            # # Ensure the source sequence length meets the requirement ORIGINAL
            # if source_enc.input_ids.shape[1] > seqlen:
            #     break  # Proceed only if the sequence fits within the seqlen
            # Ensure the source sequence length meets the requirement
            if source_enc.input_ids.shape[1] <= seqlen:
                break  # Proceed only if the sequence fits within the seqlen

        # ORIGINAL
        # # Randomly select a sequence length window within the input
        # i = random.randint(0, source_enc.input_ids.shape[1] - seqlen - 1)
        # j = i + seqlen
        # # Prepare the input and target sequences
        # inp = source_enc.input_ids[:, i:j]
        # tar = target_enc.input_ids[:, i:j].clone()
        # tar[:, :-1] = -100  # Masking all but the last token in the target sequence for language modeling
        
    # Ensure that the tokenized sequence length is at least seqlen
    if source_enc.input_ids.shape[1] >= seqlen:
        # Randomly select a sequence length window within the input
        i = random.randint(0, source_enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        # Prepare the input and target sequences
        inp = source_enc.input_ids[:, i:j]
        tar = target_enc.input_ids[:, i:j].clone()
    else:
        # If the sequence is shorter than seqlen, just take the whole sequence
        inp = source_enc.input_ids
        tar = target_enc.input_ids.clone()
        trainloader.append((inp, tar))
    tar[:, :-1] = -100  # Masking all but the last token in the target sequence for language modeling

    # Print the available fields in valdata
    print("valdata features:", valdata.features)
    for entry in valdata[:1100]:
        print(entry) 
    # source_texts = [entry['translation'][source_lang] for entry in valdata[:1100]]
    source_texts = [entry['translation'][source_lang] for entry in valdata if isinstance(entry, dict) and 'translation' in entry]
    # print(source_texts)

    joined_source_text = ' '.join(source_texts)
    # print(joined_source_text)
    
    # Tokenize the joined source texts for validation
    valenc = tokenizer(
        joined_source_text,
        return_tensors='pt', 
        max_length=(256 * seqlen), 
        truncation=True
    )

    valenc = valenc.input_ids[:, :(256 * seqlen)]

    if valenc is None:
        raise ValueError("Validation data could not be processed properly.")
    
    # valenc = tokenizer(' '.join([entry['translation'][source_lang] for entry in valdata[:1100]]), return_tensors='pt', max_length=(256 * seqlen), truncation=True)
    # valenc = tokenizer(' '.join(valdata[:1100][source_lang]), return_tensors='pt', max_length=(256 * seqlen), truncation=True)
    # valenc = valenc.input_ids[:, :(256 * seqlen)]
    
    # Wrap validation data for compatibility
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    
    traindata = load_dataset('json', data_files={'train': 'en/c4-train.00101-of-01024.json.gz'}, split='train')
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00101-of-01024.json.gz'}, split='train')
        
    # traindata = load_dataset('allenai/c4', split='train[:0.01%]')  # For Example, bc idk what the intended json files contain
    # valdata = load_dataset('allenai/c4', split='validation[:0.0001%]')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Load and process ptb dataset
def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc
    

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, source_lang="cs", target_lang="en"):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if "parallel" in name:
        return get_alma(nsamples, seed, seqlen, tokenizer, source_lang, target_lang)