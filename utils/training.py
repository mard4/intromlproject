import os
import torch
import logging
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import wandb

def train_model(model, train_loader, optimizer, criterion, epoch, model_name, dataset_name, save_dir, scheduler = None, device="cuda"):
    logger = logging.getLogger('training_logger')
    print("Training phase ...")
    model.train()
    model.to(device)
    train_loss = 0
    train_accuracy = 0
    start_time = time.time()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_accuracy += calc_accuracy(labels, torch.argmax(input = outputs, dim = 1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()
    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)
    
    # wandb.log({
    #     "epoch":epoch,
    #     "train/loss":train_loss,
    #     "train/accuracy":train_accuracy
    # })
    
    logger.info(f"Training: Average loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%", extra={'epoch': epoch})
    epoch_duration = time.time() - start_time
    logger.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds", extra={'epoch': epoch})

    return train_loss, train_accuracy

def validate_model(model, val_loader, criterion, device="cuda", epoch=None):
    print("Validation phase ...")
    model.eval()
    model.to(device)
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} - Validation" if epoch else "Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            batch_acc = calc_accuracy(labels, torch.argmax(input = outputs, dim = 1))
            val_accuracy += batch_acc

    val_loss /= len(val_loader)
    val_accuracy /= len(val_loader)
    
    # wandb.log({
    #     "epoch":epoch,
    #     "val/loss":val_loss,
    #     "val/accuracy":val_accuracy
    # })
    return val_loss, val_accuracy

def test_model(model, test_loader, device, epoch=None):
    model.eval()
    model.to(device)
    test_accuracy = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Epoch {epoch} - Testing" if epoch else "Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_accuracy += calc_accuracy(labels, outputs.argmax(dim = 1))
    test_accuracy /= len(test_loader)

    wandb.log({
        "test/accuracy":test_accuracy
    })
    return test_accuracy

def calc_accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    #print(f"Correct predictions: {correct} / {len(y_pred)}")
    return acc