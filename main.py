import json
import os
import random
from typing import Optional
from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Config
import evaluate
from pprint import pprint
from modeling_algpt2 import ALGPT2LMHeadModel

DEFAULT_MODEL_NAME = "gpt2"

# Check if Google Drive is mounted
if os.path.isdir("/content/drive"):
    save_path = "/content/drive/MyDrive/Colab\ Notebooks/AL-GPT"
else:
    save_path = "."

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_post_training(trainer: Trainer, dataset: dict, save_path: str) -> dict:
    # Evaluate the model
    trainer_evaluation_result = trainer.evaluate()
    # Compute perplexity
    perplexity = evaluate.load("perplexity", module_type="metric")
    input_texts = [s for s in dataset['test']['text'] if s != '']
    results = perplexity.compute(model_id=save_path,
                                 predictions=input_texts)
    trainer_evaluation_result['test_mean_perplexity'] = results['mean_perplexity']
    pprint(trainer_evaluation_result)
    return trainer_evaluation_result


def run(model_class_name: str, model_name: str = DEFAULT_MODEL_NAME, minimize_dataset: bool = False,
        pretrained: bool = False, depth: Optional[int] = None, batch_size: int = 32,
        num_of_epochs: float = 1.0, load_checkpoint: bool = False, dataset_path: str = "wikitext-103-raw-v1",
        sequence_max_length: int = 512, learning_rate: float = 1e-5, device="gpu", save_steps: int = 10000):
    # Load a small dataset from hugging face
    assert device.lower() in ["gpu", "tpu", "cpu"]
    assert dataset_path in ['wikitext-2-raw-v1', 'wikitext-103-raw-v1']


    dataset_path = dataset_path if not minimize_dataset else "wikitext-2-raw-v1"
    dataset = load_dataset("wikitext", dataset_path)

    if minimize_dataset:
        dataset['train'] = dataset['train'].select(range(100))
        dataset['validation'] = dataset['validation'].select(range(100))
        dataset['test'] = dataset['test'].select(range(100))
    print("train dataset size:", len(dataset['train']))
    print("validation dataset size:", len(dataset['validation']))
    print("test dataset size:", len(dataset['test']))

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the padding token for the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_class = {'GPT2LMHeadModel': GPT2LMHeadModel, 'ALGPT2LMHeadModel': ALGPT2LMHeadModel}[model_class_name]
    if pretrained:
        model = model_class.from_pretrained(model_name)
    else:
        config = GPT2Config(vocab_size=tokenizer.vocab_size) if depth is None else GPT2Config(
            vocab_size=tokenizer.vocab_size, n_layer=depth)
        model = model_class(config)
    print(model)
    print("number of parameters:", count_parameters(model))

    # Tokenize dataset
    def tokenize_function(examples):
        # Handle different datasets
        if 'text' in examples:
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=sequence_max_length)
        elif 'context' in examples and 'question' in examples:  # For datasets like 'squad_v2'
            return tokenizer(examples['context'], examples['question'], padding="max_length", truncation=True,
                             max_length=sequence_max_length)
        elif 'premise' in examples and 'hypothesis' in examples:  # For datasets like 'snli'
            return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True,
                             max_length=sequence_max_length)
        elif 'sentence' in examples:  # For datasets like 'sst2'
            return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=sequence_max_length)
        else:
            raise ValueError("Dataset structure not recognized.")

    tokenized_datasets_path = f"{save_path}/tokenized_datasets/{dataset_path}"

    if os.path.exists(tokenized_datasets_path):
        # Load tokenized_datasets from disk
        print("Loading tokenized_datasets from disk...")
        tokenized_datasets = load_from_disk(tokenized_datasets_path)
    else:
        # Tokenize dataset
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        # Add labels for the language modeling task
        tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
        # Save tokenized_datasets to disk
        tokenized_datasets.save_to_disk(tokenized_datasets_path)
        # Update model configuration
        model.config.is_decoder = True

    # shuffle the training dataset
    tokenized_datasets = tokenized_datasets.shuffle(seed=random.randint(0, 100))
        
    # Define training arguments and initialize Trainer

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=num_of_epochs,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy='steps',
        save_steps=save_steps if not minimize_dataset else 10,
        learning_rate=learning_rate,
        evaluation_strategy='steps',
        eval_steps=save_steps if not minimize_dataset else 10,
        warmup_steps=1000,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model if device.lower() != "tpu" else model.to(device),
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )

    full_path = f"{save_path}/save_{model_class_name}-{depth}-{dataset_path}"
    # Start training
    trainer.train(resume_from_checkpoint=full_path) if load_checkpoint else trainer.train()


    # Save the model
    trainer.save_model(full_path)
    trainer_evaluation_result = evaluate_post_training(trainer, dataset, full_path)
    with open(f"{full_path}/eval_results.json", 'w') as f:
        json.dump(trainer_evaluation_result, f)


if __name__ == '__main__':
    run(model_class_name='GPT2LMHeadModel', minimize_dataset=True, pretrained=False, depth=3, load_checkpoint=False,
        num_of_epochs=0.5,
        dataset_path="wikitext-2-raw-v1")
