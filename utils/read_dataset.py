import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from scipy.io import loadmat


def read_dataset_path(img_root):
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

def read_dataset(img_root, transform):
    """
    Loads and transforms the dataset using torchvision's ImageFolder.
    Splits the dataset into train, validation, and test sets.

    Args:
        img_root (str): The root directory containing the 'train', 'val', and 'test' subdirectories.
        transform (torchvision.transforms.Compose): The transformations to apply to the dataset images.

    Returns:
        tuple: A tuple containing the train, validation, and test datasets.
    """
    train_root, val_root, test_root = read_dataset_path(img_root)

    train_dataset = datasets.ImageFolder(root=train_root, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_root, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_root, transform=transform)
    
    print("Read dataset successfully")
    return train_dataset, val_dataset, test_dataset

def get_data_loader(train_dataset, val_dataset, test_dataset, batch_size):
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

def plot_firstimg(dataset):
    """
    Plots the first image in the dataset along with its class label.

    Args:
        dataset (torchvision.datasets.ImageFolder): The dataset from which to plot the first image.

    Returns:
        None
    """
    class_labels = dataset.classes
    first_image, first_label = dataset[0]

    npimg = first_image.numpy()  # Convert the image tensor to numpy array for visualization

    # Unnormalize the image (invert the normalization applied in the transform)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    npimg = np.transpose(npimg, (1, 2, 0))  # Reorder dimensions for matplotlib (128, 128, 3)
    npimg = npimg * std + mean  # Unnormalize

    # Display the image using matplotlib with its corresponding class label
    plt.imshow(npimg)
    plt.title(f"Label: {class_labels[first_label]}")
    plt.axis('off')  # Turn off axis
    plt.show()
    return

def transform_dataset(resize=(256,256), crop=(224,224),
                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    
    Args:
        resize (tuple, optional): _description_. Defaults to (256,256).
        crop (tuple, optional): _description_. Defaults to (224,224).
        mean (list, optional): _description_. Defaults to [0.485, 0.456, 0.406].
        std (list, optional): _description_. Defaults to [0.229, 0.224, 0.225].

    Default are based on ImageNet dataset
    Returns:
        transform object
    """
    transform = transforms.Compose([
    transforms.Resize(resize),                                                
    transforms.RandomCrop(crop), 
    #transforms.Resize((128, 128)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)  
    ])

    return transform

def reorganize_dataset_txt(dataset_path, txt_path, new_dataset_path, data_type):
    """
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