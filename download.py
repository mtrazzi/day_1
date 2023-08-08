from transformers import AutoModel, AutoConfig, AutoTokenizer
import os

def download_huggingface_model(model_name, save_dir="."):
    """
    Downloads all files related to a Hugging Face model to a specified directory.
    
    Args:
    - model_name (str): Name or path of the model on the Hugging Face Model Hub.
    - save_dir (str): Directory where the model will be saved. Default is the current directory.
    """

    # Make sure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Downloading model weights
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_dir)

    # Downloading model configuration
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(save_dir)

    # Downloading tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    model_name = "bert-base-uncased"  # Replace with the model name or path of your choice
    save_directory = "./downloaded_model"  # Replace with your preferred save location
    download_huggingface_model(model_name, save_directory)
