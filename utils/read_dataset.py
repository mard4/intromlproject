import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

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

# def augment_dataset(img_root):
#     return 

def split_dataset(dataset):
    """Split dataset into train, validation, and test sets (e.g., 70% train, 10% validation, 20% test)"""
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

def read_dataset_path(img_root):
    train_root = f"{img_root}/train"
    val_root = f"{img_root}/val"
    test_root = f"{img_root}/test"
    return train_root, val_root, test_root


def read_dataset(img_root, transform):
    """
    Transform and load dataset using torchvision.datasets.ImageFolder
    Split the dataset into train, validation, and test sets (e.g., 70% train, 10% validation, 20% test)
    """
    train_root, val_root, test_root = read_dataset_path(img_root)

    #transform = transform_dataset(img_root)
    train_dataset = datasets.ImageFolder(root=train_root, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_root, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_root, transform=transform)
    print("Read dataset successfully")
    #train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    return train_dataset, val_dataset, test_dataset 

def get_data_loader(train_dataset, val_dataset, test_dataset, batch_size):
    """Create DataLoader for train, validation, and test sets"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("Data loader successfully")
    return train_loader, val_loader, test_loader

def plot_firstimg(dataset):
    """Check the number of classes in the train dataset"""
    #num_classes_train = len(dataset.dataset.classes)
    #print("Number of classes in train dataset:", num_classes_train)

    class_labels = dataset.classes
    # Access the first image and its label for visualization
    first_image, first_label = dataset[0]

    # Convert the image tensor to numpy array for visualization
    npimg = first_image.numpy()  # Shape: (3, 128, 128)

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
