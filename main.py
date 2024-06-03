import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import torchvision
import torchvision.models as models
import wandb
from utils.main.data_loading import get_data_loaders, get_data_loaders_comp, get_test_loader_comp, get_label_ids
from utils.main.models_init import init_model, load_checkpoint
from utils.main.criterion import init_criterion
from utils.main.optimizers import init_optimizer
from utils.main.scheduler import init_scheduler
from utils.main.training import train_model, evaluate_model
from utils.main.exam import submit, test_model
from utils.main.optuna import get_data_loaders_optuna, run_optuna

print("Imports done")

# Load config file
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

# Some order and other variables
config['num_classes'] = len(os.listdir(config['data_dir'] + '/train'))
model = init_model(model_name = config['model_name'],num_classes = config['num_classes'], dropout_rate = config['dropout_rate']).to(config['device'])
criterion = init_criterion(config['criterion'])
optimizer = init_optimizer(model, config)
scheduler = init_scheduler(optimizer, config)
# Load checkpoint model if specified
if config['checkpoint'] is not None:
    model = load_checkpoint(model, config['checkpoint'], config['device'])
    print("Checkpoint loaded correctly")

print("Parameters loaded")

print("\nRun info:")
print("Model name: ", config['model_name'], model)
print("Dataset: ", config['dataset_name'])
print("Batch size: ", config['batch_size'])
print("Number of epochs: ", config['epochs'])
print("Number of classes: ", config['num_classes'])
print("Learning Rate: ", config['learning_rate'])
print("Dropout Rate: ", config['dropout_rate'])
print("Weight Decay: ", config['weight_decay'])
print("Momentum: ", config['momentum'])
print("Criterion: ", config['criterion'], criterion)
print("Optimizer: ", config['optimizer'], optimizer)
print("Scheduler: ", config['scheduler'], scheduler)
print("Scheduler Step Size: ", config['step_size'])
print("Scheduler Gamma: ", config['gamma'])
print("Patience: ", config['patience'])
print("Resize: ", config['resize'])
print("Crop size: ", config['crop_size'])
print("Mean: ", config['mean'])
print("Std: ", config['std'])

# Name for wandb
name = config['dataset_name'] + 'train' if config['train'] == True else config['dataset_name']
name = name + 'test' if config['test'] == True else name
name = name + 'Optuna' if config['optuna'] == True else name

# Initialize wandb
wandb.login() 
wandb.init(project=config['project_name'],
    name = name,
    config=config
)
print("\nWandb initialized")

if config['optuna']:
    train_loader, val_loader = get_data_loaders_optuna(config)
    print("Optuna data loaded")

    best_params = run_optuna(config, model, train_loader, val_loader, criterion)

    model.add_module("dropout", nn.Dropout(p=best_params['dropout_rate']))
    print(config['optimizer'].lower())
    if best_params['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=best_params['learning_rate'], 
                              weight_decay=best_params['weight_decay'], 
                              momentum=best_params['momentum'])
    elif best_params['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'],
                                 weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'],
                               weight_decay=best_params['weight_decay'])
    else:
        raise ValueError("Optimizer not implemented")
    print("Optuna done, parameters loaded")

    wandb.log({
        "new best parameters":best_params,
        "config":config
    })

    print("New best parameters:")
    print("Dropout rate: ", best_params['dropout_rate'])
    print("Optimizer: ", optimizer)
    print("Learning rate: ", best_params['learning_rate'])
    print("Weight decay: ", best_params['weight_decay'])
    print("Momentum: ", best_params['momentum'])

# Load data
if not config['comp']:
    train_loader, val_loader, test_loader = get_data_loaders(config['data_dir'], config['batch_size'], config['resize'],
                                                             config['crop_size'], config['mean'], config['std'])
    print("Data loaded")
else:
    if config['dataset_name'] != 'comp':
        raise ValueError("Dataset name must be 'comp' for competition data")
    train_loader, val_loader = get_data_loaders_comp(config['data_dir'], config['batch_size'], config['resize'], 
                                                     config['crop_size'], config['mean'], config['std'])
    print("Data loaded")

if config['train']:
    start = time.time()
    print("\nStart training!\n")

    # Train model
    model = train_model(model = model, train_loader = train_loader, val_loader = val_loader, criterion = criterion, 
                        optimizer = optimizer, scheduler = scheduler, num_epochs = config['epochs'], device = config['device'],
                        patience = config['patience'], model_name = config['model_name'], dataset_name = config['dataset_name'], 
                        save_dir = config['save_dir'])

    train_time = time.time() - start
    print(f"\nEnd training!\nTraining time: {train_time} seconds")
    wandb.log({
            "training_time": f"{(train_time):.3} seconds",
            })

if config['test'] and not config['comp']:

    # Evaluate model
    accuracy, loss = evaluate_model(model, test_loader, criterion = criterion, device = config['device'])
    print('Test Accuracy: {:.4f}'.format(accuracy))
    wandb.log({
            "test/loss":loss,
            "test/accuracy":accuracy,
        })
    print("Testing done!")

elif config['test'] and config['comp']:  

    print("Testing on competition data means submit results!")
    test_loader = get_test_loader_comp(config['data_dir'], config['resize'], config['crop'], 
                                       config['mean'], config['std'])
    model.eval()
    label_ids = get_label_ids(os.path.join(config['data_dir'], 'train'))
    preds = test_model(model, test_loader, label_ids, config['device'])
    res = {
        "images": preds,
        "groupname": "Angel Warrior of the Balls"
    }

    submit(res)
    print("Submission done!")