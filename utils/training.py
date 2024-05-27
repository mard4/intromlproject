import os
import torch
import logging
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import wandb

def train_one_epoch(model, train_loader, val_loader, optimizer, cost_function, epoch, model_name, dataset_name, save_dir, scheduler = None, device="cuda"):
    """
    Train and validate the model for one epoch, update the logs, and save the model weights.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): The optimizer.
        cost_function (torch.nn.Module): The loss function.
        epoch (int): The current epoch number.
        model_name (str): The name of the model.
        dataset_name (str): The name of the dataset.
        save_dir (str): Directory to save the logs and model weights.
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler.
        device (str): The device to run the training on (default is "cuda").

    Returns:
        tuple: A tuple containing training loss, training accuracy, validation loss, and validation accuracy.
    """
    logger = logging.getLogger('training_logger')
    model.train()
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    start_time = time.time()

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = cost_function(outputs, targets)
        loss.backward()
        optimizer.step()

        samples += inputs.size(0)
        cumulative_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        cumulative_accuracy += predicted.eq(targets).sum().item()

    train_loss = cumulative_loss / samples
    train_accuracy = cumulative_accuracy / samples * 100

    logger.info(f"Training: Average loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%", extra={'epoch': epoch})

    if val_loader is not None:
        val_loss, val_accuracy = validate(model, val_loader, cost_function, device, epoch)
        logger.info(f"Validation: Average loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%", extra={'epoch': epoch})
    else:
        val_loss, val_accuracy = None, None

    if scheduler is not None:
        scheduler.step()

    # Log metrics to wandb
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    })

    # Ensure the save directory exists
    save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_epoch{epoch}.pth")

    os.makedirs(save_dir, exist_ok=True)

    # Save the model weights
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved as {save_path}", extra={'epoch': epoch})

    # Log epoch duration
    epoch_duration = time.time() - start_time
    logger.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds", extra={'epoch': epoch})

    return train_loss, train_accuracy, val_loss, val_accuracy

def validate(model, val_loader, cost_function, device="cuda", epoch=None):
    """
    Validate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        cost_function (torch.nn.Module): The loss function.
        device (str): The device to run the validation on (default is "cuda").
        epoch (int, optional): The current epoch number for logging with tqdm.

    Returns:
        tuple: A tuple containing validation loss and validation accuracy.
    """
    model.eval()
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # Use tqdm to visualize the validation progress
    for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch} - Validation" if epoch else "Validation"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = cost_function(outputs, targets)

        samples += inputs.size(0)
        cumulative_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)

        cumulative_accuracy += predicted.eq(targets).sum().item()

    val_loss = cumulative_loss / samples
    val_accuracy = cumulative_accuracy / samples * 100

    return val_loss, val_accuracy
