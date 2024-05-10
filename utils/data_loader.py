import os
import zipfile
import tarfile
import kaggle
from PIL import Image, ImageOps

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

