import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def reorganize_dataset_txt(dataset_path, txt_path, new_dataset_path, data_type):
    """
    This function is used when the data is a folder of unorganized images and a .txt files that maps images to classes.
    Reorganizes a dataset into a new directory structure.

    The new structure has separate directories for training, validation, and testing data.
    Each of these directories contains one subdirectory for each class, and each class
    subdirectory contains the images for that class.

    Parameters:
    dataset_path (str): The path to the original dataset.
    txt_path (str): The path to a text file containing image IDs and class labels.
    new_dataset_path (str): The path where the new dataset structure should be created.
    data_type (str): The type of data ('train', 'val', or 'test').

    Returns:
    None
    """
    # Create the new dataset directory if it doesn't exist
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # Open the text file and read the lines
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Iterate over the lines in the file
    for line in lines:
        # Each line is expected to be in the format 'image_id class_label1 class_label2'
        image_id, rest = line.strip().split(' ', 1)
        class_label = rest.replace(' ', '_')

        # Create a new directory for the class if it doesn't exist
        new_class_dir = os.path.join(new_dataset_path, data_type, class_label)
        if not os.path.exists(new_class_dir):
            os.makedirs(new_class_dir)

        # Copy the image to the new directory
        old_image_path = os.path.join(dataset_path, image_id + '.jpg')
        new_image_path = os.path.join(new_class_dir, image_id + '.jpg')
        shutil.copyfile(old_image_path, new_image_path)



def reorganize_dataset_mat(dataset_path, mat_path, new_dataset_path, data_type):
    """
    This function is used when the data is a folder of unorganized images and a .csv files that maps images to classes.

    Reorganizes a dataset into a new directory structure.

    The new structure has separate directories for training, validation, and testing data.
    Each of these directories contains one subdirectory for each class, and each class
    subdirectory contains the images for that class.

    Parameters:
    dataset_path (str): The path to the original dataset.
    mat_path (str): The path to a .mat file containing image labels.
    new_dataset_path (str): The path where the new dataset structure should be created.
    data_type (str): The type of data ('train', 'val', or 'test').

    Returns:
    None
    """
    # Create the new dataset directory if it doesn't exist
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)

    # Load the .mat file
    mat = loadmat(mat_path)
    labels = mat['labels'][0]

    # Iterate over the labels
    for i, label in enumerate(labels):
        # Create a new directory for the class if it doesn't exist
        new_class_dir = os.path.join(new_dataset_path, data_type, str(label))
        if not os.path.exists(new_class_dir):
            os.makedirs(new_class_dir)

        # Copy the image to the new directory
        old_image_path = os.path.join(dataset_path, 'image_{:05d}.jpg'.format(i+1))
        new_image_path = os.path.join(new_class_dir, 'image_{:05d}.jpg'.format(i+1))
        shutil.copyfile(old_image_path, new_image_path)

def cleanup(folder_path):
    """
    Deletes all subfolders and files within the specified folder that are not named 'train', 'test', or 'val'.
    
    Parameters:
    folder_path (str): The path to the folder to be cleaned.
    
    The function iterates through each item in the specified folder path. If the item is a directory and 
    its name is not 'train', 'test', or 'val', it deletes the directory and all its contents. Similarly, 
    if the item is a file and its name is not 'train', 'test', or 'val', it deletes the file. 
    
    Example usage:
    delete_unwanted_items('/path/to/your/folder')
    
    Note:
    Use this function with caution, as it will permanently delete files and folders that are not named 
    'train', 'test', or 'val'.
    """
    # Define the allowed folder and file names
    allowed_names = {'train', 'test', 'val'}
    
    # Iterate through each item in the specified folder path
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            if item not in allowed_names:
                # Delete the directory and all its contents
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item_path}")
            else:
                print(f"Skipped folder: {item_path}")
        # Check if the item is a file
        elif os.path.isfile(item_path):
            if item not in allowed_names:
                # Delete the file
                os.remove(item_path)
                print(f"Deleted file: {item_path}")
            else:
                print(f"Skipped file: {item_path}")

def create_train_val_test_folders(base_dir, train_size=0.65, val_size=0.15, test_size=0.2):
    """
    This function is used when the data is a folder + subfolders named after the classes.
    Splits images into training, validation, and test sets and organizes them into separate subdirectories.
    
    This function takes a base directory containing subdirectories, each corresponding to a class. All images in these
    class directories are then split into training, validation, and test sets according to specified proportions.
    The images are moved into new subdirectories within the base directory, organized by train, validation, and test sets.

    Parameters:
    - base_dir (str): The path to the base directory containing subdirectories of classes with images.
    - train_size (float): The proportion of the dataset to include in the train split (default is 0.6).
    - val_size (float): The proportion of the dataset to include in the validation split (default is 0.2).
    - test_size (float): The proportion of the dataset to include in the test split (default is 0.2).

    Returns:
    None

    Example:
    >>> base_dir = '/path/to/your/dataset'
    >>> create_train_val_test_folders(base_dir)

    Note:
    - The function assumes that the sum of `train_size`, `val_size`, and `test_size` equals 1.0.
    - The directories 'train', 'val', and 'test' will be created under each class directory in the base directory.
    - It is assumed that the base directory is structured such that each subdirectory represents a class and contains images.
    """
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
    print("Create train_val_test folders successfully")

def create_train_val_test_path(img_root):
    """
    Constructs the paths for the train, validation, and test datasets.

    Args:
        img_root (str): The root directory containing the 'train', 'val', and 'test' subdirectories.

    Returns:
        tuple: A tuple containing the paths for the train, validation, and test datasets.
    """
    train_root = f"{img_root}/train"
    val_root = f"{img_root}/val"
    test_root = f"{img_root}/test"
    return train_root, val_root, test_root

def create_folder_path(img_root, folder):
    """
    Constructs the paths for the folder given in input.

    Args:
        img_root (str): The root directory containing the 'train', 'val', and 'test' subdirectories.
        folder (str): The folder to be created.
    Returns:
        str: The path for the folder given in input.
    """
    final_folder = f"{img_root}/{folder}"

    return final_folder

def transform_train_val_test(img_root, transform):
    """
    Loads and transforms the dataset using torchvision's ImageFolder.
    Splits the dataset into train, validation, and test sets.

    Args:
        img_root (str): The root directory containing the 'train', 'val', and 'test' subdirectories.
        transform (torchvision.transforms.Compose): The transformations to apply to the dataset images.

    Returns:
        tuple: A tuple containing the train, validation, and test datasets.
    """
    train_root, val_root, test_root = create_train_val_test_path(img_root)
    train_dataset = datasets.ImageFolder(root=train_root, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_root, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_root, transform=transform)
    print("Read dataset successfully")
    return train_dataset, val_dataset, test_dataset

def transform_folder(folder, transform):
    """
    Applies the transformation to a folder

    Args:
        img_root (str): The root directory containing the 'train', 'val', and 'test' subdirectories.
        transform (torchvision.transforms.Compose): The transformations to apply to the dataset images.

    Returns:
        tuple: A tuple containing the train, validation, and test datasets.
    """
    dataset = datasets.ImageFolder(root=folder, transform=transform)
    print("Read dataset successfully")
    return dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Creates DataLoaders for the train, validation, and test datasets.

    Args:
        train_dataset (torchvision.datasets.ImageFolder): The training dataset.
        val_dataset (torchvision.datasets.ImageFolder): The validation dataset.
        test_dataset (torchvision.datasets.ImageFolder): The test dataset.
        batch_size (int): The number of samples per batch to load.

    Returns:
        tuple: A tuple containing the train, validation, and test DataLoaders.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print("Data loader successfully")
    return train_loader, val_loader, test_loader


def create_dataloader(folder, batch_size):
    """
    Creates DataLoaders for the provided.

    Args:
        train_dataset (torchvision.datasets.ImageFolder): The training dataset.
        val_dataset (torchvision.datasets.ImageFolder): The validation dataset.
        test_dataset (torchvision.datasets.ImageFolder): The test dataset.
        batch_size (int): The number of samples per batch to load.

    Returns:
        tuple: A tuple containing the train, validation, and test DataLoaders.
    """

    dataset = DataLoader(root=folder, batch_size=batch_size)
    print("Data loader successfully")
    return dataset



def final_structure(path, new_folder_name, folder1, folder2, folder3):
    """
    Creates a new folder and moves three other folders into it.

    Parameters:
    path (str): The path where the new folder should be created.
    new_folder_name (str): The name of the new folder.
    folder1 (str): The path to the first folder to move.
    folder2 (str): The path to the second folder to move.
    folder3 (str): The path to the third folder to move.

    Returns:
    None
    """
    # Create the new folder
    new_folder_path = os.path.join(path, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    # Move the folders into the new folder
    shutil.move(folder1, new_folder_path)
    shutil.move(folder2, new_folder_path)
    shutil.move(folder3, new_folder_path)