from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import argparse

import random
import matplotlib.pyplot as plt
import os 

if os.path.isdir("/content/drive"):
    save_path = "/content/drive/MyDrive/Colab\ Notebooks/AL-GPT"
else:
    save_path = "."

def run_tokenizer_training(dataset_path: str = "wikitext-103-raw-v1"):
    # 1. Load the wikitext-103-raw dataset
    dataset = load_dataset('wikitext', dataset_path)

    # create dir if not existing
    if not os.path.exists(f"{save_path}/tokenizer"):
        os.makedirs(f"{save_path}/tokenizer")
    # Combine the train, validation, and test splits into a single text file
    with open(f"{save_path}/tokenizer/{dataset_path}.txt", "w", encoding="utf-8") as f:
        for split in ['train', 'validation', 'test']:
            f.write("\n".join(dataset[split]['text']))

    # 2. Define the tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    # 3. Train the tokenizer on wikitext-103-raw
    trainer = trainers.BpeTrainer(vocab_size=50257, special_tokens=[
        "<|endoftext|>",
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    tokenizer.train(files=[f"{save_path}/tokenizer/{dataset_path}.txt"], trainer=trainer)

    # 4. Save the tokenizer
    tokenizer.save(f"{save_path}/tokenizer/{dataset_path}_tokenizer.json", pretty=True)




def _check_tokenizer(dataset_path: str = "wikitext-2-raw-v1"):
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{save_path}/tokenizer/{dataset_path}_tokenizer.json",)
    dataset = load_dataset('wikitext', dataset_path)
    tokenized_datasets = {}
    for split, split_dataset in dataset.items():
        tokenized_data = fast_tokenizer(split_dataset['text'])
        tokenized_datasets[split] = tokenized_data

    # 1. Print Random Samples
    for _ in range(5):
        idx = random.randint(0, len(dataset['train']) - 1)
        print("Original:", dataset['train']['text'][idx])
        print("Tokenized:", tokenized_datasets['train']['input_ids'][idx])
        print("Decoded:", fast_tokenizer.decode(tokenized_datasets['train']['input_ids'][idx]))
        print("------")

    # 2. Token Length Distribution
    token_lengths = [len(x) for x in tokenized_datasets['train']['input_ids']]
    plt.hist(token_lengths, bins=50)
    plt.xlabel('Token Length')
    plt.ylabel('Number of Samples')
    plt.title('Token Length Distribution')
    plt.show()

    # 3. Special Tokens Check
    for idx in range(5):
        if tokenized_datasets['train']['input_ids'][idx]:
            print("Start Token:", tokenized_datasets['train']['input_ids'][idx][0])  # Should be <s>
            print("End Token:", tokenized_datasets['train']['input_ids'][idx][-1])  # Should be </s>
            print("------")

    # 4. OOV Tokens Check
    oov_count = sum([1 for sample in tokenized_datasets['train']['input_ids'] for token in sample if token == fast_tokenizer.unk_token_id])
    total_tokens = sum([len(sample) for sample in tokenized_datasets['train']['input_ids']])
    oov_percentage = (oov_count / total_tokens) * 100
    print(f"Percentage of OOV tokens: {oov_percentage:.2f}%")



if __name__ == '__main__':
    _check_tokenizer()
    # parser = argparse.ArgumentParser(description="Run the tokenizer training.")
    # parser.add_argument("--dataset_path", type=str, default="wikitext-103-raw-v1", help="Path to the dataset.")
    # args = parser.parse_args()
    # run_tokenizer_training(args.dataset_path)