import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import wandb
import yaml
import optuna
import optuna.visualization
import random
from torch.optim.lr_scheduler import StepLR
from utils.logger import *
from utils.models_init import *
from utils.training2 import *  # Corrected import
from utils.optimizers import *
import torch.nn as nn
import timm

# Configuration
trials = 4  # Number of trials for Optuna
epochs_optuna = 2  # Number of epochs for Optuna trials

# Load configuration
with open('intromlproject/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

config = config['config']
config['data_dir'] = config['data_dir'].format(root=config['root'], img_folder=config['img_folder'])
config['save_dir'] = config['save_dir'].format(root=config['root'], model_name=config['model_name'], img_folder=config['img_folder'])
if config['checkpoint'] is not None:
    config['checkpoint'] = config['checkpoint'].format(root=config['root'])
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
config['project_name'] = config['project_name'].format(model_name=config['model_name'])
config['dataset_name'] = config['dataset_name'].format(img_folder=config['img_folder'])

##### DO NOT EDIT #####
print(config)
def main(config):
    # Initialize wandb
    wandb.init(project=config['project_name'],
               name=f"{config['model_name']}_{config['dataset_name']}_optuna",
               config=config)

    # Setup logger
    logger = setup_logger(log_dir=config['save_dir'])

    def load_dataloader(config):
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

        # Limit training data for faster epochs
        n_samples = config['batch_size'] * 30
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        #train_subset = Subset(train_dataset, indices[:n_samples])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        #print(train_loader.dataset)

        n_samples_val = config['batch_size'] * 10
        #val_subset = Subset(val_dataset, list(range(n_samples_val)))
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

        return train_dataset, train_loader, val_dataset, val_loader
    
    train_dataset, train_loader, val_dataset, val_loader = load_dataloader(config)
    num_classes = len(train_loader.dataset.classes)
    #print(f"Number of classes: {config['num_classes']}")
    criterion = getattr(torch.nn, config['criterion'])()

    def objective(trial):
        model = init_model(config['model_name'], num_classes).to(config['device'])
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
        model.add_module("dropout", torch.nn.Dropout(p=dropout_rate))

        optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        if optimizer_name == 'SGD':
            momentum = trial.suggest_float('momentum', 0.5, 0.99)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=0.1) if config['scheduler'] else None

        for epoch in range(1, epochs_optuna):
            train_loss, train_acc = train_model(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
                model_name=config['model_name'],
                dataset_name=config['dataset_name'],
                save_dir=config['save_dir'],
                scheduler=scheduler,
                device=config['device']
            )
            val_loss, val_acc = validate_model(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=config['device']
            )
            print(f"Train loss: {train_loss:.3f}, Train accuracy: {train_acc:.3f}")
            print(f"Validation loss: {val_loss:.3f}, Validation accuracy: {val_acc:.3f}")
            save_path = os.path.join(config['checkpoint_path'], f"OPTUNA_trial{trial}_{config['model_name']}_{config['dataset_name']}_epoch{epoch}.pth")
            os.makedirs(config['save_dir'], exist_ok=True)
            # Save the model weights
            torch.save(model.state_dict(), save_path)
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return val_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials, timeout=600)

    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    optimization_history_plot = optuna.visualization.plot_optimization_history(study)
    param_importance_plot = optuna.visualization.plot_param_importances(study)
    param_importance_plot.show()
    optimization_history_plot.show()

    ###################################### TRAIN THE MODEL WITH THE BEST HYPERPARAMETERS FOUND WITH OPTUNA ################################################



    model = init_model(config['model_name'],num_classes).to(config['device'])
    model.add_module("dropout", torch.nn.Dropout(p=best_params['dropout_rate']))

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

    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=0.1) if config['scheduler'] else None
    print("Train the model with the best hyperparameters found with Optuna...", best_params)
    print("Config",config)


if __name__ == "__main__":
    main(config)


#Trial 1 finished with value: 77.75595238095237 and parameters: {'dropout_rate': 0.33143040558969483, 'optimizer': 'SGD', 'learning_rate': 0.03110465473824186, 'weight_decay': 4.983197115094068e-05, 'momentum': 0.6908541771514911}. Best is trial 0 with value: 79.90476190476187.