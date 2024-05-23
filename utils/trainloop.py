import torch
from tqdm import tqdm
import logging

import torch
import logging

def test(net, data_loader, cost_function, device="cuda"):
    logger = logging.getLogger('training_logger')
    net.eval()
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = cost_function(outputs, targets)

            samples += inputs.size(0)
            cumulative_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()

    average_loss = cumulative_loss / samples
    accuracy = cumulative_accuracy / samples * 100

    logger.info(f"Test set: Average loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return average_loss, accuracy

def train_one_epoch(net, train_loader, val_loader, optimizer, cost_function, device="cuda"):
    logger = logging.getLogger('training_logger')
    net.train()
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cost_function(outputs, targets)
        loss.backward()
        optimizer.step()

        samples += inputs.size(0)
        cumulative_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
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
    logger = logging.getLogger('training_logger')
    net.eval()
    val_samples = 0.0
    val_cumulative_loss = 0.0
    val_cumulative_accuracy = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = cost_function(outputs, targets)

            val_samples += inputs.size(0)
            val_cumulative_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_cumulative_accuracy += predicted.eq(targets).sum().item()

    val_loss = val_cumulative_loss / val_samples
    val_accuracy = val_cumulative_accuracy / val_samples * 100

    logger.info(f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    return val_loss, val_accuracy
