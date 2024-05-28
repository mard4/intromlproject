import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import logging 
from utils.testing import *
from utils.logger import setup_logger
from utils.models_init import init_model, load_checkpoint
import yaml

with open('intromlproject/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
config = config['config']
config['data_dir'] = config['data_dir'].format(root=config['root'], img_folder=config['img_folder'])
config['checkpoint'] = config['checkpoint'].format(root=config['root'])
config['save_dir'] = config['save_dir'].format(root=config['root'], model_name=config['model_name'], img_folder=config['img_folder'])
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
config['project_name'] = config['project_name'].format(model_name=config['model_name'])
config['dataset_name'] = config['dataset_name'].format(img_folder=config['img_folder'])

def main(config):
    wandb.init(project=config['project_name'],
               name=f"TEST_{config['model_name']}_{config['dataset_name']}_batch_size: {config['batch_size']}")

    logger = setup_logger(log_dir=config['save_dir'])

    model = init_model(config['model_name'], config['num_classes'])
    model.to(config['device'])

    if config['checkpoint']:
        model = load_checkpoint(model, config['checkpoint'], config['device'])

    criterion = getattr(torch.nn, config['criterion'])()

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    test_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Use the new test function with exam submission
    test_loss, test_accuracy = test_model_exam(model, test_loader, criterion, config['device'])
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main(config)
