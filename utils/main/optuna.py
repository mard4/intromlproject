from torchvision import transforms, datasets
import random
from torch.utils.data import DataLoader, Subset
import os
import torch
import optuna
from torch.optim.lr_scheduler import StepLR
from utils.main.training import train_model

def get_data_loaders_optuna(config):
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
    train_indices = list(range(len(train_dataset)))
    random.shuffle(train_indices)
    train_subset = Subset(train_dataset, train_indices[:n_samples])
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    # Limit validation data for faster epochs
    n_samples_val = config['batch_size'] * 10
    val_indices = list(range(len(val_dataset)))
    random.shuffle(val_indices)
    val_subset = Subset(val_dataset, val_indices[:n_samples_val])
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    return train_loader, val_loader

def objective(trial, model, train_loader, val_loader, criterion, config):
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    model.add_module("dropout", torch.nn.Dropout(p=dropout_rate))

    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD', 'Adam'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.5, 0.99)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=0.1) if config['scheduler'] else None

    val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=config['opt_epochs'],
        model_name=config['model_name'],
        dataset_name=config['dataset_name'],
        save_dir=config['save_dir'],
        device=config['device'],
        patience=config['patience'],
        trial=trial,
        optuna=True
    )

    return val_acc


def run_optuna(config, model, train_loader, val_loader, criterion):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model, train_loader, val_loader, criterion, config), n_trials=config['trials'], timeout=600)

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

    return best_params
