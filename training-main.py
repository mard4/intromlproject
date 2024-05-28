import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import yaml
from torch.optim.lr_scheduler import StepLR
from utils.logger import *
from utils.models_init import *
from utils.training import *
from utils.optimizers import *
from utils.custom_models import *

#configuration file
with open('./config.yaml', 'r') as file:
    config = yaml.safe_load(file)

config = config['config']
config['data_dir'] = config['data_dir'].format(root=config['root'], img_folder=config['img_folder'])
config['save_dir'] = config['save_dir'].format(root=config['root'], model_name=config['model_name'], img_folder=config['img_folder'])
if config['checkpoint'] is not None:
    config['checkpoint'] = config['checkpoint'].format(root=config['root'])
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
config['project_name'] = config['project_name'].format(model_name=config['model_name'])
config['dataset_name'] = config['dataset_name'].format(img_folder=config['img_folder'])

###################### DO NOT EDIT #################

def main(config):
    # Initialize wandb
    wandb.init(project=config['project_name'],
               name = config['dataset_name'],
               #name=f"{config['model_name']}_{config['dataset_name']}_opt: {config['optimizer']}_batch_size: {config['batch_size']}_lr: {config['learning_rate']}",
               config=config)

    # Setup logger
    logger = setup_logger(log_dir=config['save_dir'])

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomCrop(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std'])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], 'val'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    num_classes = len(train_loader.dataset.classes)
    # Initialize the model
    model = init_model(config['model_name'],num_classes)
    model.to(config['device'])
    
    logger.info(f"Configurations: {config}")
    
    optimizer = create_optimizer(model, config)

    # Define the loss function
    criterion = getattr(torch.nn, config['criterion'])()

    # Define the learning rate scheduler
    if config['scheduler'] == True:
        scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=0.1)
    else:
        scheduler = None

    # Load checkpoint if specified
    if config['checkpoint']:
        model = load_checkpoint(model, config['checkpoint'], config['device'])
        print("Checkpoint loaded correctly")

    # Early stopping
    counter = 0
    patience = config['patience']
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, config['epochs'] + 1):

        train_loss, train_accuracy, val_loss, val_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            cost_function=criterion,
            epoch=epoch,
            model_name=config['model_name'],
            dataset_name=config['dataset_name'],
            save_dir=config['save_dir'],
            scheduler=scheduler,
            device=config['device']
        )
        print(f"Epoch {epoch} completed. Train Loss: {train_loss}, Train Acc: {train_accuracy}, Val Loss: {val_loss}, Val Acc: {val_accuracy}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Early stopping...')
                break

if __name__ == "__main__":
    main(config)