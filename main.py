import json
import os
import random
import wandb
from typing import Optional
from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Config, TrainerCallback, \
    TrainerState, TrainerControl, PreTrainedTokenizerFast, BertLMHeadModel, BertConfig, BertTokenizerFast
from pprint import pprint
from modeling_algpt2 import ALGPT2LMHeadModel
import math

DEFAULT_MODEL_NAME = "gpt2"



class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None,
                    **kwargs):
        if metrics is None:
            metrics = {}
        if 'eval_loss' in metrics:
            # Calculate perplexity from the eval loss and add it to metrics
            metrics['eval_perplexity'] = math.exp(metrics['eval_loss'])

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Access the logs, which should contain 'loss'
        logs = kwargs['logs']
        if 'loss' in logs:
            # Calculate perplexity from the train loss and add it to logs
            logs['train_perplexity'] = math.exp(logs['loss'])
            wandb.log(logs)


# Check if Google Drive is mounted
if os.path.isdir("/content/drive"):
    save_path = "/content/drive/MyDrive/Colab\ Notebooks/AL-GPT"
else:
    save_path = "."


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_post_training(trainer: Trainer, dataset: dict) -> dict:
    # Evaluate the model
    trainer_evaluation_result = trainer.evaluate(eval_dataset=dataset['test'])
    # Compute perplexity
    trainer_evaluation_result['test_mean_perplexity'] = math.exp(trainer_evaluation_result['eval_loss'])
    pprint(trainer_evaluation_result)
    return trainer_evaluation_result


def run(model_class_name: str, model_name: str = DEFAULT_MODEL_NAME, minimize_dataset: bool = False,
        pretrained: bool = False, depth: Optional[int] = None, batch_size: int = 32,
        num_of_epochs: float = 1.0, load_checkpoint: bool = False, dataset_path: str = "wikitext-103-raw-v1",
        sequence_max_length: int = 512, learning_rate: float = 1e-5, device="gpu", save_steps: int = 10000,
        tokenizer_path: Optional[str] = None):
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
    if tokenizer_path is None or tokenizer_path == '':  # Use the default tokenizer
        tokenizer_object = BertTokenizerFast if model_class_name == 'BertLMHeadModel' else GPT2Tokenizer
        tokenizer = tokenizer_object.from_pretrained(model_name)
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{save_path}/tokenizer/{tokenizer_path}_tokenizer.json", )

    # Set the padding token for the tokenizer
    # if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model_class = {'GPT2LMHeadModel': GPT2LMHeadModel,
                   'ALGPT2LMHeadModel': ALGPT2LMHeadModel,
                   'BertLMHeadModel': BertLMHeadModel}[model_class_name]
    model_config = {'GPT2LMHeadModel': GPT2Config,
                    'ALGPT2LMHeadModel': GPT2Config,
                    'BertLMHeadModel': BertConfig}[model_class_name]
    if pretrained:
        model = model_class.from_pretrained(model_name)
    else:
        config = model_config(vocab_size=tokenizer.vocab_size) if depth is None else GPT2Config(
            vocab_size=tokenizer.vocab_size, n_layer=depth)
        model = model_class(config)
    print(model)
    print("number of parameters:", count_parameters(model))

    if not load_checkpoint and not pretrained:
        model.init_weights()

    # Tokenize dataset
    def tokenize_function(examples):
        # Handle different datasets
        return tokenizer(examples['text'], padding="max_length", truncation=True,
                         max_length=sequence_max_length, return_attention_mask=True)

    tokenized_datasets_path = f"{save_path}/tokenized_datasets/{dataset_path}"

    if os.path.exists(tokenized_datasets_path):
        # Load tokenized_datasets from disk
        print("Loading tokenized_datasets from disk...")
        tokenized_datasets = load_from_disk(tokenized_datasets_path, keep_in_memory=True)
    else:
        # Tokenize dataset
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        # Add labels for the language modeling task
        tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
        # Save tokenized_datasets to disk
        tokenized_datasets.save_to_disk(tokenized_datasets_path)

    # shuffle the training dataset
    # tokenized_datasets = tokenized_datasets.shuffle(seed=random.randint(0, 100))

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
        eval_steps=1000 if not minimize_dataset else 10,
        warmup_steps=0,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        callbacks=[PerplexityCallback()],
    )

    full_path = f"{save_path}/save_{model_class_name}-{depth}-{dataset_path}"
    # Start training
    trainer.train(resume_from_checkpoint=full_path) if load_checkpoint else trainer.train()

    # Save the model
    trainer.save_model(full_path)
    trainer_evaluation_result = evaluate_post_training(trainer, dataset)
    with open(f"{full_path}/eval_results.json", 'w') as f:
        json.dump(trainer_evaluation_result, f)


if __name__ == '__main__':
    run(model_class_name='GPT2LMHeadModel', minimize_dataset=True, pretrained=False, depth=3, load_checkpoint=False,
        num_of_epochs=0.5,
        dataset_path="wikitext-2-raw-v1")
