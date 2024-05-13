import os
import tarfile
import kaggle
from PIL import Image, ImageOps
import shutil
import pandas as pd
# Downloaders

def download_dataset_kaggle(dataset_url, target_folder):
    """
    Downloads a dataset from Kaggle given a URL and saves it to a specified folder.
    
    Args:
    dataset_url (str): URL of the Kaggle dataset. It should follow this form: https://www.kaggle.com/datasets/wenewone/cub2002011
    target_folder (str): Path to the folder where the dataset should be downloaded. "." downloads everything to the current folder. 
    It is recommended as it allows you to have everything working from the beginning.
    
    Returns:
    str: Path to the downloaded dataset directory.
    """
    # Extracting the dataset name and owner from the URL
    path_parts = dataset_url.split('/')
    dataset_name = path_parts[-1].split('?')[0]
    owner = path_parts[-2]

    # Create target directory if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Constructing Kaggle API dataset download command
    kaggle_dataset = f'{owner}/{dataset_name}'
    
    # Changing the working directory to the target folder
    os.chdir(target_folder)

    # Using the Kaggle API to download the dataset
    # Make sure Place the kaggle.json file in the location ~/.kaggle/kaggle.json on Unix-based systems or 
    # C:\Users\<Windows-username>\.kaggle\kaggle.json on Windows. 
    # This allows the API client to automatically recognize your credentials.
    # Make sure to set its permissions like this "chmod 600 ~/.kaggle/kaggle.json" (unix)

    kaggle.api.dataset_download_files(kaggle_dataset, path=target_folder, unzip=True)
    return target_folder

def extract_tgz(tgz_path, extract_to=None):
    """
    Extracts a .tgz file to a specified directory.

    Args:
    tgz_path (str): The file path of the .tgz file to be extracted.
    extract_to (str): Optional. The directory where files will be extracted.
                      If not specified, extracts to the directory containing the .tgz file.

    Returns:
    str: The path where the files were extracted.
    """
    if extract_to is None:
        extract_to = os.path.dirname(tgz_path)

    # Ensure the extraction directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Open the tgz file and extract it
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    
    print(f"Files have been extracted to: {extract_to}")
    return extract_to

# Example usage:
# extract_path = extract_tgz('/path/to/your/file.tgz')


def get_data_file(folder_name, dataset_url):
    """Download and extract a dataset, returning the local directory path."""
    if not dataset_url:
        raise ValueError("Dataset URL cannot be empty.")
    return tf.keras.utils.get_file(
        folder_name, 
        origin=dataset_url, 
        cache_dir='.', 
        untar=True
    )

def resize_images_in_folder(root_folder, width, height):
    """
    Resizes all images in the specified folder and its subfolders to the given width and height.
    
    Args:
    root_folder (str): The path to the root folder containing images.
    width (int): The new width for the images.
    height (int): The new height for the images.
    """
    # Supported image formats
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    # Walk through all directories and files in the root folder
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(extensions):
                image_path = os.path.join(subdir, file)
                with Image.open(image_path) as img:
                    # Resize the image using the high-quality LANCZOS filter
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    # Save the resized image back to the same location
                    img.save(image_path)
                    print(f'Resized and saved: {image_path}')

