import os
from typing import Optional
from datasets import load_dataset
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Config
from pprint import pprint
from modeling_algpt2 import ALGPT2LMHeadModel


DEFAULT_MODEL_NAME = "gpt2"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run(model_class_name: str, model_name: str = DEFAULT_MODEL_NAME, minimize_dataset: bool = False, pretrained: bool = False, depth: Optional[int] = None, batch_size: int = 32):
    # Load a small dataset from hugging face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1") # ['squad_v2', 'sst2', 'snli', 'openwebtext', 'wikitext-2']
    print("dataset size:", len(dataset['train']))
    if minimize_dataset:
        dataset['train'] = dataset['train'].select(range(100))
        dataset['validation'] = dataset['validation'].select(range(100))

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the padding token for the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_class = {'GPT2LMHeadModel': GPT2LMHeadModel, 'ALGPT2LMHeadModel': ALGPT2LMHeadModel}[model_class_name]
    if pretrained:
        model = model_class.from_pretrained(model_name)
    else:
        config = GPT2Config(vocab_size=tokenizer.vocab_size) if depth is None else GPT2Config(vocab_size=tokenizer.vocab_size, n_layer=depth)
        model = model_class(config)
    print(model)
    print("number of parameters:", count_parameters(model))

    # Tokenize dataset
    def tokenize_function(examples):
        # Handle different datasets
        if 'text' in examples:  
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
        elif 'context' in examples and 'question' in examples:  # For datasets like 'squad_v2'
            return tokenizer(examples['context'], examples['question'], padding="max_length", truncation=True, max_length=128)
        elif 'premise' in examples and 'hypothesis' in examples:  # For datasets like 'snli'
            return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True, max_length=128)
        elif 'sentence' in examples:  # For datasets like 'sst2'
            return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=128)
        else:
            raise ValueError("Dataset structure not recognized.")
        
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Add labels for the language modeling task
    tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': examples['input_ids']}, batched=True)

    # Update model configuration
    model.config.is_decoder = True

    # Define training arguments and initialize Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100000,
        eval_steps=100000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

    # Evaluate the model
    result = trainer.evaluate()
    pprint(result)

    # Check if Google Drive is mounted
    if os.path.isdir("/content/drive"):
        save_path = "/content/drive/MyDrive/Colab\ Notebooks/AL-GPT"
    else:
        save_path = "."
    # Save the model
    trainer.save_model(f"{save_path}/save_{model_class_name}-{depth}")
    

if __name__ == '__main__':
    run(model_class_name='GPT2LMHeadModel', minimize_dataset=False, pretrained=False, depth=4)