from utils.download_data import *
from utils.epochs_loop import *
from utils.logger import *
from utils.init_models import *
from utils.read_dataset import *
import torch
import torch.nn as nn
import os

# !!! IMPORTANT: DO NOT DELETE ANY OF THE VARIABLES BELOW, JUST EDIT THE VALUES !!!

"""
FOLDERS: 
        - Soil types
        - CUB
"""

img_root = "/home/disi/ml/intromlproject/datasets/Aerei"
# Standardize the path
standardized_path = os.path.normpath(img_root)

# Split the path into components
path, folder = os.path.split(standardized_path)

print("Path:", path)
print("Folder:", folder)
"""
MODELS: 
        alexnet, efficientnet, inceptionv4, inceptionv4_freeze, inceptionv3, inceptionv3_freeze, densenet201
"""
model_name = "densenet201"
"""
Training size for each model:
        - alexnet: 224
        - efficientnet: 224
        - inceptionv4: 299
        - inceptionv4_freeze: 299
        - inceptionv3: 299
        - inceptionv3_freeze: 299
        - densenet201: 224
"""

transform = transform_dataset(
    resize=(256, 256),
    crop=(224, 224),
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

learning_rate = 0.001
weight_decay = 0.000001
momentum = 0.9
betas = (0.9, 0.999)
epochs = 10
criterion = nn.CrossEntropyLoss()
batch_size = 60
optimizer_type = "custom"  # "simple" or "custom"
optimz = torch.optim.Adam
param_groups = [{'prefixes': ['classifier'], 'lr': learning_rate * 10},
                {'prefixes': ['features']}]

#########################################  DO NOT EDIT BELOW THIS LINE  #########################################
log_file_path = os.path.join(path, 'training.log')
logger = setup_logger(log_file_path)

logger.info("Starting the training process")

train_data, val_data, test_data = read_dataset(img_root, transform=transform)
num_classes = len(train_data.classes)
model = init_model(model_name, num_classes)

optimizer = get_optimizer(
    optimizer_type=optimizer_type,
    model=model,
    lr=learning_rate,
    wd=weight_decay,
    momentum=momentum,
    betas=betas,
    param_groups=param_groups if optimizer_type == "custom" else None,
    optim=optimz
)

train_loader, val_loader, test_loader = get_data_loader(train_data, val_data, test_data, batch_size)

main(
    run_name=f"{model_name}_{folder}_{optimizer_type}_{str(optimz)}",
    batch_size=batch_size,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_path=f'{path}models/checkpoint_{model_name}.pth',
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    momentum=momentum,
    betas=betas,
    epochs=epochs,
    criterion=criterion,
    optimizer_type=optimizer_type,
    optim=optimz,
    param_groups=param_groups if optimizer_type == "custom" else None,
    visualization_name=f"{model_name}",
    img_root=img_root,
    save_every=1,
    init_model=model,
    transform=transform,
    val_loader=val_loader  # Pass the validation loader, can be None
)

logger.info("Training process finished")
# Uncomment to start TensorBoard automatically
# %tensorboard --logdir logs/fit
