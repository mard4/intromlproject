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
from utils.training import *  # Corrected import
from utils.optimizers import *
import torch.nn as nn
import timm

# Configuration
trials = 5  # Number of trials for Optuna
epochs_optuna = 10  # Number of epochs for Optuna trials

# Load configuration
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

train_transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std'])
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(config['data_dir'], 'val'), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

num_classes = len(train_loader.dataset.classes)
print("num classes", num_classes)
model = init_model(config['model_name'], num_classes)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc

def objective(trial, model, train_loader, val_loader, config):
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_val_acc = 0
    for epoch in range(epochs_optuna):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs_optuna}")
        print(f"Train loss: {train_loss:.3f}, Train accuracy: {train_acc:.3f}")
        print(f"Validation loss: {val_loss:.3f}, Validation accuracy: {val_acc:.3f}")
        
        save_path = os.path.join(config['save_dir'], f"OPTUNA_trial{trial.number}_{config['model_name']}_{config['dataset_name']}_epoch{epoch}.pth")
        os.makedirs(config['save_dir'], exist_ok=True)
        # Save the model weights
        torch.save(model.state_dict(), save_path)
        
        trial.report(val_acc, epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, model, train_loader, val_loader, config), n_trials=trials, timeout=600)

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
