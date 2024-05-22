import torch
from tqdm import tqdm
import logging

def test(net, data_loader, cost_function, device="cuda"):
    """
    Evaluate the model on the test dataset.

    Args:
        net (torch.nn.Module): The neural network to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        cost_function (torch.nn.Module): The loss function.
        device (str, optional): Device to run the evaluation on. Default is "cuda".

    Returns:
        tuple: Average loss and accuracy of the model on the test dataset.
    """
    logger = logging.getLogger('training_logger')
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    net.eval()  # Set the network to evaluation mode

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net(inputs)
            loss = cost_function(outputs, targets)

            samples += inputs.shape[0]
            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()

    logger.info(f"Test set: Average loss: {cumulative_loss / samples:.4f}, Accuracy: {cumulative_accuracy / samples * 100:.2f}%")
    return cumulative_loss / samples, cumulative_accuracy / samples * 100

def train_one_epoch(net, train_loader, val_loader, optimizer, cost_function, device="cuda"):
    """
    Train the model for one epoch and optionally validate on the validation set.

    Args:
        net (torch.nn.Module): The neural network to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader or None): DataLoader for the validation dataset. Can be None.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        cost_function (torch.nn.Module): The loss function.
        device (str, optional): Device to run the training on. Default is "cuda".

    Returns:
        tuple: Training loss, training accuracy, validation loss, and validation accuracy.
               If val_loader is None, validation loss and accuracy will be None.
    """
    logger = logging.getLogger('training_logger')
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    net.train()  # Set the network to training mode

    for _, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net(inputs)
        loss = cost_function(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        cumulative_accuracy += predicted.eq(targets).sum().item()

    train_loss = cumulative_loss / samples
    train_accuracy = cumulative_accuracy / samples * 100

    logger.info(f"Training: Average loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    if val_loader is not None:
        val_loss, val_accuracy = validate(net, val_loader, cost_function, device)
        logger.info(f"Validation: Average loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    else:
        val_loss, val_accuracy = None, None

    return train_loss, train_accuracy, val_loss, val_accuracy

def validate(net, val_loader, cost_function, device="cuda"):
    """
    Validate the model on the validation dataset.

    Args:
        net (torch.nn.Module): The neural network to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        cost_function (torch.nn.Module): The loss function.
        device (str, optional): Device to run the validation on. Default is "cuda".

    Returns:
        tuple: Average loss and accuracy of the model on the validation dataset.
    """
    logger = logging.getLogger('training_logger')
    net.eval()  # Set the network to evaluation mode

    val_samples = 0.0
    val_cumulative_loss = 0.0
    val_cumulative_accuracy = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net(inputs)
            loss = cost_function(outputs, targets)

            val_samples += inputs.shape[0]
            val_cumulative_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            val_cumulative_accuracy += predicted.eq(targets).sum().item()

    val_loss = val_cumulative_loss / val_samples
    val_accuracy = val_cumulative_accuracy / val_samples * 100

    logger.info(f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    return val_loss, val_accuracy
