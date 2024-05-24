import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

from utils.logger import setup_logger
from utils.models_init import init_model, load_checkpoint
from utils.training import validate

"""
Image Sizes:
    - AlexNet: 224x224
    - DenseNet: 224x224
    - Inception: 299x299
    - ResNet: 224x224
    - VGG: 224x224

Mean, Std, number of classes for Datasets:
    - CUB-200-2011: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=200
    - Stanford Dogs: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=120
    - FGVC Aircraft: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=102
    - Flowers102: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=102
"""

# Configuration
config = {
    # Path and directory stuff
    'data_dir': '/home/disi/machinelearning/datasets',  # Directory containing the dataset
    'dataset_name': 'aerei',  # Name of the dataset you are using, doesn't need to match the real name, just a word to distinguish it
    'checkpoint': '/home/disi/machinelearning/checkpoints/alexnet/alexnet_aerei_epoch2.pth',  # Path to a checkpoint file to load
    'save_dir': '/home/disi/machinelearning/checkpoints/alexnet',  # Directory to save logs and model checkpoints
    'project_name': 'alexnet_test',  # Weights and Biases project name
    
    # Image transformation 
    'image_size': 224,  # Size of the input images (default: 224)
    'num_classes': 102,  # Number of classes in the dataset
    'mean': [0.485, 0.456, 0.406],  # Mean for normalization
    'std': [0.229, 0.224, 0.225],  # Standard deviation for normalization

    # Testing loop
    'model_name': 'alexnet',  # Name of the model to use
    'batch_size': 32,  # Batch size (default: 32)
    'criterion': 'CrossEntropyLoss',  # Criterion for the loss function (default: CrossEntropyLoss)

    # Irrelevant
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use for testing


## non modificare ua sotto

}
def main(config):
    """
    This script evaluates a trained model on the test dataset and logs the results.

    Args:
    - config (dict): Configuration dictionary containing parameters such as:
        - data_dir (str): Directory containing the dataset
        - save_dir (str): Directory to save logs and model checkpoints
        - project_name (str): Weights and Biases project name
        - model_name (str): Name of the model to use
        - num_classes (int): Number of classes in the dataset
        - checkpoint (str): Path to a checkpoint file to load
        - batch_size (int): Batch size for data loaders
        - image_size (int): Size of the input images
        - device (str): Device to use for testing (e.g., 'cuda' or 'cpu')
        - mean (list): Mean values for normalization
        - std (list): Standard deviation values for normalization
        - criterion (str): Criterion for the loss function

    Returns:
    None
    """
    # Initialize wandb
    wandb.init(project=config['project_name'])

    # Setup logger
    logger = setup_logger(log_dir=config['save_dir'])

    # Initialize the model
    model = init_model(config['model_name'], config['num_classes'])
    model.to(config['device'])

    # Modify the classifier layer to match the current dataset's number of classes
    if config['model_name'] == 'alexnet':
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, config['num_classes'])

    # Load checkpoint
    if config['checkpoint']:
        model = load_checkpoint(model, config['checkpoint'], config['device'])

    # Define the loss function
    criterion = getattr(torch.nn, config['criterion'])()

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    # Load test dataset
    test_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Validate the model on the test dataset
    test_loss, test_accuracy = validate(model, test_loader, criterion, config['device'], epoch=None)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Log results to wandb
    wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy})

if __name__ == "__main__":
    main(config)
