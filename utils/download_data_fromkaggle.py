import os
import tarfile
import kaggle
from PIL import Image, ImageOps
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile
import tensorflow as tf
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
    print("Kaggle Dataset downloaded successfully.")
    return

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

# organizer

def organize_images(csv_train, image_folder, csv_val=None):
    """
    Organizes images into subfolders based on class labels from one or two CSV files.
    It creates separate 'train' and 'val' folders if validation CSV is provided.

    Args:
    csv_train (str): Path to the CSV file containing training image names and class labels.
    image_folder (str): Path to the folder containing all the images.
    csv_val (str, optional): Path to the CSV file containing validation image names and class labels.
                             If None, only training images are organized.

    Returns:
    None: Images are moved into subfolders within the original image folder.
    """
    def move_images(csv_path, base_folder):
        # Load the CSV file into a DataFrame
        data = pd.read_csv(csv_path)

        # Loop through the rows in the DataFrame
        for index, row in data.iterrows():
            image_name, class_label = row[0], row[1]
            dest_dir = os.path.join(base_folder, str(class_label))

            # Create the directory if it doesn't exist
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            src_file = os.path.join(image_folder, image_name)
            dest_file = os.path.join(dest_dir, image_name)

            # Move the file if it exists in the source
            if os.path.exists(src_file):
                shutil.move(src_file, dest_file)

    # Base folder for training images
    train_base = os.path.join(image_folder, 'train')
    move_images(csv_train, train_base)

    if csv_val:
        # Base folder for validation images
        val_base = os.path.join(image_folder, 'val')
        move_images(csv_val, val_base)

    print(f"Images have been organized into 'train' and 'val' subfolders in '{image_folder}'.")

def create_train_val_test_folders(base_dir, train_size=0.6, val_size=0.2, test_size=0.2):
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for folder in folders:
        images = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        train, temp = train_test_split(images, train_size=train_size, test_size=(val_size+test_size), random_state=42)
        val, test = train_test_split(temp, train_size=val_size/(val_size + test_size), test_size=test_size/(val_size + test_size), random_state=42)

        # Function to move files
        def move_files(files, dest):
            for f in files:
                shutil.move(f, os.path.join(dest, os.path.basename(f)))

        # Create train, validation, test directories
        os.makedirs(os.path.join(base_dir, 'train', os.path.basename(folder)), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'val', os.path.basename(folder)), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', os.path.basename(folder)), exist_ok=True)

        # Move files
        move_files(train, os.path.join(base_dir, 'train', os.path.basename(folder)))
        move_files(val, os.path.join(base_dir, 'val', os.path.basename(folder)))
        move_files(test, os.path.join(base_dir, 'test', os.path.basename(folder)))

#extractors

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

def extract_zip(zip_path, extract_to=None):
    """
    Extracts a .zip file to a specified directory.

    Args:
    zip_path (str): The file path of the .zip file to be extracted.
    extract_to (str): Optional. The directory where files will be extracted.
                      If not specified, extracts to the directory containing the .zip file.

    Returns:
    str: The path where the files were extracted.
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    # Ensure the extraction directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Open the zip file and extract it
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Files have been extracted to: {extract_to}")
    return extract_to


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