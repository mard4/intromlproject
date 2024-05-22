import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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