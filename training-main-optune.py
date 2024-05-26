import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from utils.logger import *
from utils.models_init import *
from utils.training import *
from utils.optimizers import *

# Configuration

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
root = '/home/disi/ml'
img_folder = 'aircraft'
model_name = 'efficientnetv2'
config = {
    # Path and directory stuff
    'data_dir': f'{root}/datasets/{img_folder}',  # Directory containing the dataset
    'dataset_name' : f"{img_folder}", # Name of the dataset you are using, doesn't need to match the real name, just a word to distinguish it
    # leave checkpoint = None if you don't have one
    'checkpoint': None,#f'{root}/checkpoints/alexnet/alexnet_aerei_epoch2.pth',  # Path to a checkpoint file to resume training
    'save_dir': f'{root}/checkpoints/{model_name}/optuna',  # Directory to save logs and model checkpoints
    'project_name': f'{model_name}_test',  # Weights and Biases project name
    
    
    # Image transformation 
    'image_size': 224,  # Size of the input images (default: 224)
    'num_classes': 102,  # Number of classes in the dataset
    'mean': [0.485, 0.456, 0.406],  # Mean for normalization
    'std': [0.229, 0.224, 0.225],  # Standard deviation for normalization

    # Training loop
    'model_name': f'{model_name}',  # Name of the model to use
    'batch_size': 42,  # Batch size (default: 32)
    'epochs': 10,  # Number of epochs to train (default: 10)
    'optimizer': 'SGD',  # Optimizer to use (default: Adam)
    'optimizer_type': 'simple',  # Type of optimizer to use (default: simple)
    'learning_rate': 0.001,  # Learning rate (default: 0.001)    #0.08573324997589683
    'weight_decay': 0,  # Weight decay for optimizer (default: 0)
    'momentum': 0,  # Momentum for optimizer (default: 0)
    'criterion': 'CrossEntropyLoss',  # Criterion for the loss function (default: CrossEntropyLoss)

    #Irrelevant
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use for training
}

### NON MODIFICAR EQUA SOTTO O MARTINA SI INCAZZA (MI STA PUNTANDO U NCOLTELLO ALLA GOLA)

def main(config):
    import optuna
    # Initialize wandb
    wandb.init(project=config['project_name'],
               name=f"{config['model_name']}_{config['dataset_name']}_opt: {config['optimizer']}_batch_size: {config['batch_size']}_lr: {config['learning_rate']}",
               #sync_tensorboard=True,
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

    # Define the loss function
    criterion = getattr(torch.nn, config['criterion'])()

    # Define the objective function for Optuna
    def objective(trial):
        """ Define hyperparameters and model
            Train and evaluate the model
            Return the evaluation metric
        """
        # Define the model
        model = init_model(config['model_name'], config['num_classes']).to(config['device'])
        optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        if optimizer_name == 'SGD':
            momentum = trial.suggest_float('momentum', 0.5, 0.99)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            momentum = 0.0  # Default momentum for non-SGD optimizers
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
  

        # Training loop
        for epoch in range(1, 2):#config['epochs'] + 1):
            train_loss, train_accuracy, val_loss, val_accuracy=train_one_epoch(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                cost_function=criterion,
                epoch=epoch,
                model_name=config['model_name'],
                dataset_name=config['dataset_name'],
                save_dir=config['save_dir'],
                device=config['device']
            )
            trial.report(val_loss, epoch)
        return val_loss

    # Create the study and optimize the objective function
    study = optuna.create_study(direction='minimize')    #minimize the val loss or maximize the val accuracy (if so return the val_loss)
    study.optimize(objective, n_trials=2)
    
    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    
    import optuna.visualization as vis
    optimization_history_plot = vis.plot_optimization_history(study)
    param_importance_plot = vis.plot_param_importances(study)
    param_importance_plot.show()
    optimization_history_plot.show()
    print("trials done")

    
    ######################################################################################

    #### OPTIONAL!!!!!!!!!!!!!!! Train the model with the best hyperparameters
    model = init_model(config['model_name'], config['num_classes']).to(config['device'])
    if best_params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay'],
            momentum=best_params['momentum']
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
    )

    for epoch in range(1, config['epochs'] + 1):
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            cost_function=criterion,
            epoch=epoch,
            model_name=config['model_name'],
            dataset_name=config['dataset_name'],
            save_dir=config['save_dir'],
            device=config['device']
        )

if __name__ == "__main__":
    main(config)

