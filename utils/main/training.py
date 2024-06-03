import os
import torch
import logging
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import wandb
from torch import optim

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, patience, model_name, dataset_name, save_dir, 
                optuna=False, trial = None):

    best_val_loss = float('inf')
    best_val_acc = 0.0
    counter = 0

    for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch'):

        e_start = time.time()
        print("\n", '-'*10)
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {current_lr}')     

        if model_name == 'seresnet50' and epoch < 13:
            model.unfreeze_layer(-(epoch + 1))
            # NB seresnet has to be used with SDG optimizer
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0

        for inputs, labels in tqdm(train_loader, desc='Epoch training', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs.data, dim=1)
            running_acc += torch.eq(labels, preds).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = running_acc / total
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')
        print('Epoch time: ', time.time() - e_start)

        # Check for best validation accuracy and save model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'Best model saved with val accuracy: {best_val_acc:.4f}')
         
        if scheduler is not None:
            scheduler.step()

        wandb.log({
            "epoch":epoch+1,
            "train/loss":epoch_loss,
            "train/accuracy":train_acc,
            "val/loss":val_loss,
            "val/accuracy":val_acc,
        })

        # Check for best validation loss and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Early stopping...')
                break
        
        save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_epoch{epoch}.pth")
        os.makedirs(save_dir, exist_ok=True)
        # Save the model weights
        torch.save(model.state_dict(), save_path)
        
        if optuna:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    print('Training complete. Best validation accuracy: {:.4f}'.format(best_val_acc))

    if optuna:
        return best_val_acc
    return model


def evaluate_model(model, data_loader, criterion=None, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    if criterion is not None:
        avg_loss = running_loss / len(data_loader.dataset)
        return avg_loss, accuracy
    return accuracy

