from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments


def run():
    # Load a small dataset from hugging face
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # Load tokenizer and model
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the padding token for the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

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
        save_steps=10,
        eval_steps=10,
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