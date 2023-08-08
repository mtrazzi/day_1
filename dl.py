import os
from huggingface_hub import snapshot_download

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

    # Download the entire model repository to the specified directory
    snapshot_download(model_name, cache_dir=save_dir, force_download=True, resume_download=False)

if __name__ == "__main__":
    model_name = "TheBloke/Llama-2-7B-GGML"  # Replace with the model name or path of your choice
    save_directory = "./downloaded_model"  # Replace with your preferred save location
    download_huggingface_model(model_name, save_directory)

