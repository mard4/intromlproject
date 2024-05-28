import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import yaml
from torch.optim.lr_scheduler import StepLR
from utils.logger import *
from utils.models_init import *
from utils.training2 import *  ### correct this mardeen @edit
from utils.optimizers import *
from utils.custom_models import *

#configuration file @edit
with open('/home/disi/ml/intromlproject/config.yaml', 'r') as file:
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
def run_epochs(model,
                train_loader,
                val_loader,
                #test_loader = test_loader,
                num_epochs,
                patience,
                device,
                model_name,
                dataset_name,
                save_dir,
                checkpoint_path,
                criterion,
                optimizer,
                logger,
                scheduler = None,
                ):
    for epoch in range(1, num_epochs + 1):
        counter = 0
        patience = config['patience']
        best_val_loss = float('inf')
        train_loss, train_acc = train_model(model = model,
                                            train_loader = train_loader,
                                            optimizer = optimizer,
                                            criterion = criterion,
                                            epoch=epoch,
                                            model_name=model_name,
                                            dataset_name=dataset_name,
                                            save_dir=save_dir,
                                            scheduler = None,
                                            device = device
                                            )
        val_loss, val_acc = validate_model(model = model,
                                           val_loader = val_loader,
                                           criterion = criterion,
                                           device = device)
        wandb.log({
            "train/loss":train_loss,
            "train/accuracy":train_acc,
            "val/loss":val_loss,
            "val/accuracy":val_acc
        })
        if scheduler is not None:
             scheduler.step()
        print(f"\nTraining loss: {train_loss:.3f}, Training accuracy: {train_acc:.3f}")
        print(f"Validation loss: {val_loss:.3f}, Validation accuracy: {val_acc:.3f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Early stopping...')
                break

        
        save_path = os.path.join(checkpoint_path, f"{model_name}_{dataset_name}_epoch{epoch}.pth")
        os.makedirs(save_dir, exist_ok=True)
        # Save the model weights
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved as {save_path}", extra={'epoch': epoch})


def main(config):
    #torch.manual_seed(123)
    
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
    #test_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], 'test'), transform=transform)
    #test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    print("Data loaded correctly")
    print("number of classes",len(train_loader.dataset.classes))
    num_classes = len(train_loader.dataset.classes)
    
    # Load checkpoint model if specified
    if config['checkpoint']:
        model = load_checkpoint(model, config['checkpoint'], config['device'])
        print("Checkpoint loaded correctly")
        
    # Initialize the model
    model = init_model(config['model_name'], num_classes=num_classes)
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



    run_epochs(model = model,
                  train_loader = train_loader,
                  val_loader = val_loader,
                  #test_loader = test_loader,
                  num_epochs = config['epochs'],
                  patience = config['patience'],
                  device = config['device'],
                  model_name=config['model_name'],
                  dataset_name=config['dataset_name'],
                  save_dir=config['save_dir'],
                  checkpoint_path = config['save_dir'],
                  criterion = criterion,
                  optimizer = optimizer,
                  logger=logger,
                  scheduler = scheduler
                  )
    
    
main(config)