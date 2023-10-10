from main import run, DEFAULT_MODEL_NAME
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the model with specified parameters.")
        
    parser.add_argument("--model_class_name", type=str, help="Name of the model class.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Name of the model.")
    parser.add_argument("--minimize_dataset", action="store_true", help="Minimize the dataset.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model.")
    parser.add_argument("--depth", type=int, help="Depth of the model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_of_epochs", type=float, default=1.0, help="Number of epochs.")
    parser.add_argument("--load_checkpoint", action="store_true", help="Load from a checkpoint.")
    parser.add_argument("--dataset_path", type=str, default="wikitext-103-raw-v1", help="Path to the dataset.")
    parser.add_argument("--sequence_max_length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"], help="Device to run on.")
    parser.add_argument("--save_steps", type=int, default=10000, help="Steps interval to save the model.")
    # tokenizer_path can be None
    parser.add_argument("--tokenizer_path", type=str, default='', help="Path to the tokenizer.")
    parser.add_argument("--load_tokenized_datasets", type=bool, default=True, help="Load tokenized datasets.")


    args = parser.parse_args()

    run(model_class_name=args.model_class_name,
        model_name=args.model_name,
        minimize_dataset=args.minimize_dataset,
        pretrained=args.pretrained,
        depth=args.depth,
        batch_size=args.batch_size,
        num_of_epochs=args.num_of_epochs,
        load_checkpoint=args.load_checkpoint,
        dataset_path=args.dataset_path,
        sequence_max_length=args.sequence_max_length,
        learning_rate=args.learning_rate,
        device=args.device,
        save_steps=args.save_steps,
        tokenizer_path=args.tokenizer_path)