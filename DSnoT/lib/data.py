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

LANG_MAP = {
    'en': 'English',
    'de': 'German',
    'cs': 'Czech',
    'ru': 'Russian',
    'zh': 'Chinese',
    'is': 'Icelandic'
}

def get_alma(nsamples, seed, seqlen, tokenizer, source_lang, target_lang):
    from datasets import Dataset
    import pandas as pd
    import random

    # Define all language directions
    full_alma_splits = {
        'train': {
            'cs-en': 'cs-en/train-00000-of-00001-3a60b130a713425b.parquet', # ok
            'de-en': 'de-en/train-00000-of-00001-39460826cd7ac756.parquet', # ok 
            # 'is-en': 'is-en/train-00000-of-00001-f71a989f63b28d68.parquet', # ok
            'ru-en': 'ru-en/train-00000-of-00001-3ba3fad04eea46f0.parquet', # ok
            'zh-en': 'zh-en/train-00000-of-00001-6bd744feceb30dbf.parquet'  # ok
        },
        'validation': {
            'cs-en': 'cs-en/validation-00000-of-00001-d1f9a3fc339fbc84.parquet', # ok
            'de-en': 'de-en/validation-00000-of-00001-34198d3f975c1787.parquet', # ok
            # 'is-en': 'is-en/validation-00000-of-00001-bb3b8280f4b7ff31.parquet',
            'ru-en': 'ru-en/validation-00000-of-00001-e9c97fe731036b74.parquet', # ok
            'zh-en': 'zh-en/validation-00000-of-00001-d1cc83e30e3dcdb2.parquet'  # ok
        }
    }
   
    # Set seed for reproducibility
    random.seed(seed)

    # Prepare the data loader
    trainloader = []
    all_validation_texts = []

    # Iterate over all language directions
    for lang_pair in full_alma_splits['train'].keys():
        print(f"Loading dataset for {lang_pair}")

        # Load train and validation splits
        train_split_path = f"hf://datasets/haoranxu/ALMA-Human-Parallel/{full_alma_splits['train'][lang_pair]}"
        print("Train link: ", train_split_path)
        train_df = pd.read_parquet(train_split_path)

        # Handle the 'is-en' case by splitting off validation data
        if lang_pair == 'is-en':
            print(f"Splitting 'is-en' training data into train and validation sets")
            # Split 10% of the training data into validation
            train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=seed)
            valdata = Dataset.from_pandas(val_df)
        else:
            val_split_path = f"hf://datasets/haoranxu/ALMA-Human-Parallel/{full_alma_splits['validation'][lang_pair]}"
            print("Validation link: ", val_split_path)
            val_df = pd.read_parquet(val_split_path)
            valdata = Dataset.from_pandas(val_df)
            
        # Convert DataFrames to Hugging Face Dataset
        traindata = Dataset.from_pandas(train_df)
        valdata = Dataset.from_pandas(val_df)

        source_lang, target_lang = lang_pair.split('-')

        # Sample data for training
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                translations = traindata[i]['translation']
                source_text = translations.get(source_lang)
                target_text = translations.get(target_lang)

                if source_text is None or target_text is None:
                    continue  # Skip this entry if either source or target text is missing

                prompt = (
                    f"Translate this from {LANG_MAP[source_lang]} to {LANG_MAP[target_lang]}:\n"
                    f"{LANG_MAP[source_lang]}: {source_text}\n"
                    f"{LANG_MAP[target_lang]}:"
                )

                # Tokenize source and target text
                source_enc = tokenizer(prompt, return_tensors='pt', max_length=seqlen, truncation=True)
                target_enc = tokenizer(target_text, return_tensors='pt', max_length=seqlen, truncation=True)

                if source_enc.input_ids.shape[1] <= seqlen:
                    break

            # Prepare the input and target sequences for training
            inp = source_enc.input_ids
            tar = target_enc.input_ids.clone()
            trainloader.append((inp, tar))
        
        # Collect validation data
        source_texts = [entry['translation'][source_lang] for entry in valdata if 'translation' in entry]
        all_validation_texts.extend(source_texts)

        # Handle inverted language pairs (e.g., en-cs, en-de, en-ru, etc.)
        inverted_lang_pair = f"{target_lang}-{source_lang}"
        print(f"Handling inverted pair: {inverted_lang_pair}")

        # Sample data by inverting source and target
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                translations = traindata[i]['translation']
                source_text = translations.get(target_lang)  # Inverted: use target as source
                target_text = translations.get(source_lang)  # Inverted: use source as target

                if source_text is None or target_text is None:
                    continue  # Skip this entry if either source or target text is missing

                prompt = (
                    f"Translate this from {LANG_MAP[target_lang]} to {LANG_MAP[source_lang]}:\n"
                    f"{LANG_MAP[target_lang]}: {source_text}\n"
                    f"{LANG_MAP[source_lang]}:"
                )
                # Tokenize source and target text (inverted)
                source_enc = tokenizer(prompt, return_tensors='pt', max_length=seqlen, truncation=True)
                target_enc = tokenizer(target_text, return_tensors='pt', max_length=seqlen, truncation=True)

                if source_enc.input_ids.shape[1] <= seqlen:
                    break

            # Prepare the input and target sequences for training (inverted)
            inp = source_enc.input_ids
            tar = target_enc.input_ids.clone()
            trainloader.append((inp, tar))

    # Join all validation texts from multiple languages
    joined_source_text = ' '.join(all_validation_texts)


    # Tokenize validation data
    valenc = tokenizer(
        joined_source_text,
        return_tensors='pt', 
        max_length=(256 * seqlen), 
        truncation=True
    )
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    if valenc is None:
        raise ValueError("Validation data could not be processed properly.")

    # Wrap validation data for compatibility
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc



# def get_alma_test(seqlen, tokenizer, source_lang, target_lang):
#     import pandas as pd
#     from datasets import Dataset

#     # Load ALMA test dataset
#     test_df = pd.read_parquet("hf://datasets/haoranxu/WMT22-Test/cs-en/test-00000-of-00001-1a83a591805d9178.parquet")

#     # Convert the Pandas DataFrame back into a Hugging Face Dataset
#     testdata = Dataset.from_pandas(test_df)

#     # Prepare test data: Join source texts into a single string and tokenize
#     source_texts = [entry['translation'][source_lang] for entry in testdata]
#     joined_source_text = ' '.join(source_texts)

#     # Tokenize the test data
#     testenc = tokenizer(joined_source_text, return_tensors='pt', max_length=seqlen, truncation=True)

#     return testenc

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