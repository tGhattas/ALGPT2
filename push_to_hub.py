import os
from typing import Optional

import argparse

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from modeling_algpt2 import ALGPT2LMHeadModel

MODEL_CLASSES = {
    'GPT2LMHeadModel': GPT2LMHeadModel,
    'ALGPT2LMHeadModel': ALGPT2LMHeadModel
}


def push_model_to_hub(model_class_name: str, depth: Optional[int] = None, dataset_path: str = "wikitext-2-v1",
                      save_path=".", factorized_embeds: bool = False, hf_name_appendix: str = ""):
    # Assert that model_class_name is valid
    assert model_class_name in MODEL_CLASSES, f"Model class {model_class_name} not supported. Supported classes are: {list(MODEL_CLASSES.keys())}"

    # Define a unique model name for Hugging Face Model Hub
    model_name = f"{model_class_name}-{depth}-{dataset_path}"

    # Define where your trained model and tokenizer are saved
    full_path = f"{save_path}/save_{model_name}"

    # Check if model exists in the path
    if not os.path.exists(os.path.join(full_path, 'pytorch_model.bin')):
        raise ValueError(f"No model found in {full_path}. Make sure the path is correct.")

    # Load your trained model
    model = MODEL_CLASSES[model_class_name].from_pretrained(full_path, factorized_embeds=factorized_embeds)

    # Load your trained tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(full_path)

    # Push model and tokenizer to Hugging Face Model Hub
    hf_model_name = f"{model_class_name}-{depth if depth is not None else 'default_depth'}-{dataset_path}"
    appendix = f"_{hf_name_appendix}" if hf_name_appendix != "" else ""
    hf_model_name = f"{hf_model_name}_{'factorized_embeds' if factorized_embeds else 'not_factorized_embeds'}{appendix}"
    model.push_to_hub(hf_model_name, use_auth_token=True)
    tokenizer.push_to_hub(hf_model_name, use_auth_token=True)

    print(f"Model and tokenizer have been pushed to Hugging Face Model Hub with name: {model_name}")


if __name__ == "__main__":
    # Example usage:
    # push_model_to_hub(model_class_name="ALGPT2LMHeadModel", depth=2, dataset_path="wikitext-2-v1",
    #                   save_path=".", factorized_embeds=True)

    parser = argparse.ArgumentParser(description="Push a trained model and tokenizer to Hugging Face Model Hub")
    parser.add_argument("--model_class_name", required=True, choices=MODEL_CLASSES.keys(),
                        help="Name of the model class. E.g., ALGPT2LMHeadModel")
    parser.add_argument("--depth", type=int, default=None,
                        help="Depth of the model (e.g., number of layers). E.g., 12 or None for default")
    parser.add_argument("--dataset_path", default="wikitext-2-v1",
                        help="Path of the dataset used. E.g., wikitext-2-v1")
    parser.add_argument("--save_path", default=".",
                        help="Path where the trained model and tokenizer are saved. E.g., ./path_to_your_saved_model_directory")
    parser.add_argument("--factorized_embeds", default=False, action="store_true", help="is factorized_embeds")
    parser.add_argument("--hf_name_end", default="",
                        help="Appendix to the hugging face name with underscore")
    args = parser.parse_args()

    # Run the function with the provided arguments
    push_model_to_hub(model_class_name=args.model_class_name, depth=args.depth, dataset_path=args.dataset_path,
                      save_path=args.save_path, factorized_embeds=args.factorized_embeds, hf_name_appendix=args.hf_name_end)
