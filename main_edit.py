import torch
from efficientnet_pytorch import EfficientNet
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from utils.init_models import *
from utils.init_checkpoints import *
from utils.trainloop import *
from utils.read_dataset import *

def get_optimizer(model, lr, wd, momentum):
    """specifica per alexnet da modificare"""
    # We will create two groups of weights, one for the newly initialized layer
    # and the other for rest of the layers of the network
    final_layer_weights = []
    rest_of_the_net_weights = []

    # Iterate through the layers of the network
    for name, param in model.named_parameters():
        if name.startswith('classifier.6'):
            final_layer_weights.append(param)
        else:
            rest_of_the_net_weights.append(param)

    # Assign the distinct learning rates to each group of parameters
    optimizer = torch.optim.SGD([
        {'params': rest_of_the_net_weights},
        {'params': final_layer_weights, 'lr': lr}
    ], lr=lr / 10, weight_decay=wd, momentum=momentum)

    return optimizer

def main(run_name,
        batch_size,
        device,
        checkpoint_path,
        learning_rate,
        weight_decay,
        momentum,
        epochs,
        criterion,
        visualization_name,
        img_root,
        save_every,
        init_model,
        transform):
    
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
        
    train_dataset, val_dataset, test_dataset = read_dataset(img_root, transform)
    trainloader, valloader, testloader = get_data_loader(train_dataset, val_dataset, test_dataset, batch_size)
    plot_firstimg(train_dataset)

    # Instantiates the model
    num_classes = len(train_dataset.classes)
    #net = initialize_model(num_classes).to(device)
    net = init_model.to(device)
    
    # cost function
    #optimizer = optim.Adam(net.parameters(), lr=0.001)
    optimizer = get_optimizer(net, learning_rate, weight_decay, momentum)
    
    # Range over the number of epochs
    pbar = tqdm(range(epochs), desc="Training", position=0, leave=True)
    for e in range(epochs):
        # Train and log
        train_loss, train_accuracy, val_loss, val_accuracy = train_one_epoch(net, trainloader,valloader, optimizer, criterion)

        test_loss, test_accuracy = test(net, testloader, criterion)
        
        # checkpoint
        checkpoint_perepoch(e, save_every, net, optimizer, train_loss, train_accuracy, val_accuracy, checkpoint_path)   # correct with validation accyraacy

        # Add values to plots
        writer.add_scalar("train/loss", train_loss, e + 1)
        writer.add_scalar("train/accuracy", train_accuracy, e + 1)
        writer.add_scalar("test/loss", test_loss, e + 1)
        writer.add_scalar("test/accuracy", test_accuracy, e + 1)

        pbar.set_postfix(train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)
        pbar.update(1)

    # Compute and print final metrics
    print("After training:")
    train_loss, train_accuracy = test(net, trainloader, criterion)
    test_loss, test_accuracy = test(net, testloader, criterion)

    print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    
    final_checkpoint(e, net, optimizer, train_loss, train_accuracy, test_accuracy, test_accuracy, checkpoint_path)   # correct with validation accyraacy
    
    # Add values to plots
    writer.add_scalar("train/loss", train_loss, epochs + 1)
    writer.add_scalar("train/accuracy", train_accuracy, epochs + 1)
    writer.add_scalar("test/loss", test_loss, epochs + 1)
    writer.add_scalar("test/accuracy", test_accuracy, epochs + 1)

    # Close the logger
    writer.close()

    return net, optimizer

# data_path = "./data/CUB/"
# folder_train = "birds_train"
# img_root = data_path + folder_train

# main("alexnet_sgd_0.01_RW", img_root=img_root)
# %tensorboard --logdir logs/fit