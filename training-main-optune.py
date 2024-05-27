import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
import yaml
from utils.logger import *
from utils.models_init import *
from utils.training import *
from utils.optimizers import *
import optuna

# Configuration

root = '/home/disi/ml'
img_folder = 'aircraft'
model_name = 'vit'
trials = 3   # Number of trials for Optuna
epochs_optuna = 1  # Number of epochs for Optuna trials
with open('intromlproject/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

config = config['config']
config['data_dir'] = config['data_dir'].format(root=config['root'], img_folder=config['img_folder'])
config['save_dir'] = config['save_dir'].format(root=config['root'], model_name=config['model_name'], img_folder=config['img_folder'])
config['checkpoint'] = config['checkpoint'].format(root=config['root'])
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
config['project_name'] = config['project_name'].format(model_name=config['model_name'])
config['dataset_name'] = config['dataset_name'].format(img_folder=config['img_folder'])

##### DO NOT EDIT #####

def main(config):
    
    # Initialize wandb
    wandb.init(project=config['project_name'],
               name=f"{config['model_name']}_{config['dataset_name']}_optuna",
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
    print(f"Number of classes: {config['num_classes']}")
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
        model.add_module("dropout", torch.nn.Dropout(p=0.5))  # Add dropout layer

        optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        
        if optimizer_name == 'SGD':
            momentum = trial.suggest_float('momentum', 0.5, 0.99)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            momentum = 0.0  # Default momentum for non-SGD optimizers
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Define the learning rate scheduler
        if config['scheduler'] == True:
            scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=0.1)
        else:
            scheduler = None

        # Training loop
        for epoch in range(epochs_optuna+1):#config['epochs'] + 1):
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
        return val_accuracy

    # Create the study and optimize the objective function
    study = optuna.create_study(direction='maximize')    #minimize the val loss or maximize the val accuracy (if so return the val_loss)
    study.optimize(objective, n_trials=trials)
    
    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    
    import optuna.visualization as vis
    optimization_history_plot = vis.plot_optimization_history(study)
    param_importance_plot = vis.plot_param_importances(study)
    param_importance_plot.show()
    optimization_history_plot.show()
    print("trials done")

    
    ###################################### TRAIN THE MODEL WITH THE BEST HYPEPARAMTERS FOUND WITH OPTUNA ################################################

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
    # Define the learning rate scheduler
    if config['scheduler'] == True:
        scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=0.1)
    else:
        scheduler = None

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
            save_dir=config[f'{root}/checkpoints/{model_name}/optuna'],
            scheduler = scheduler,
            device=config['device']
        )

if __name__ == "__main__":
    main(config)

