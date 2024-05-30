import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import yaml
from utils.logger import setup_logger
from utils.models_init import init_model, load_checkpoint
from utils.training import test_model

#configuration file
with open('intromlproject/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
config = config['config']
config['data_dir'] = config['data_dir'].format(root=config['root'], img_folder=config['img_folder'])
if config['checkpoint'] is not None:
    config['checkpoint'] = config['checkpoint'].format(root=config['root'])
config['save_dir'] = config['save_dir'].format(root=config['root'], model_name=config['model_name'], img_folder=config['img_folder'])
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
config['project_name'] = config['project_name'].format(model_name=config['model_name'])
config['dataset_name'] = config['dataset_name'].format(img_folder=config['img_folder'])

###################### DO NOT EDIT #################
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
    wandb.init(project=config['project_name'],
               name=f"TEST_{config['model_name']}_{config['dataset_name']}_batch_size: {config['batch_size']}")

    # Setup logger
    logger = setup_logger(log_dir=config['save_dir'])

    # Initialize the model
    train_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], 'train'), transform=transform)
    train_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    num_classes = len(train_loader.dataset.classes)
    model = init_model(config['model_name'], num_classes=num_classes)

    # Load checkpoint
    if config['checkpoint']:
        model = load_checkpoint(model, config['checkpoint'], config['device'])
    
    model.to(config['device'])

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
    test_loss, test_accuracy = test_model(model, test_loader, criterion, config['device'], epoch=None)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Log results to wandb
    wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy})

if __name__ == "__main__":
    main(config)