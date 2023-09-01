from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

from modeling_algpt2 import ALGPT2LMHeadModel


def run():
    # Load a small dataset from hugging face
    dataset = load_dataset('squad_v2') # ['squad_v2', 'sst2', 'snli', 'openwebtext']

    # Load tokenizer and model
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the padding token for the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ALGPT2LMHeadModel.from_pretrained(model_name)

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
        
    # Minimize dataset for faster experimentation
    dataset['train'] = dataset['train'].shuffle(seed=42).select(range(1000))

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Add labels for the language modeling task
    tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': examples['input_ids']}, batched=True)

    # Update model configuration
    model.config.is_decoder = True

    # Define training arguments and initialize Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
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

    trainer.evaluate()

    print(len(tokenized_datasets["train"]))

if __name__ == '__main__':
    run()