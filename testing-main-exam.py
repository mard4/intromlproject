import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import logging 
from utils.testing import *
from utils.logger import setup_logger
from utils.models_init import init_model, load_checkpoint

root = '/home/disi/ml'
img_folder = 'fiori'
model_name = 'efficientnetv2'
checkpoint_pth = f'efficientnetv2_fiori_epoch10.pth'

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


config = {
    'data_dir': f'{root}/datasets/{img_folder}',
    'dataset_name': img_folder,
    'checkpoint': f'{root}/checkpoints/{model_name}_{img_folder}/{checkpoint_pth}',
    'save_dir': f'{root}/checkpoints/{model_name}',
    'project_name': f'{model_name}_test',
    'image_size': 224,
    'num_classes': 102,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'model_name': model_name,
    'batch_size': 16,
    'criterion': 'CrossEntropyLoss',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

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
