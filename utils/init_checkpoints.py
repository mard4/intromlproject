import torch
import os
from datetime import datetime


def checkpoint_perepoch(epoch, save_every, net, optimizer, train_loss, train_accuracy, val_accuracy, checkpoint_dir):
    if epoch % save_every == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        }

        # Ensure the checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Create a unique filename for each checkpoint
        checkpoint_filename = f'checkpoint_epoch_{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")


def final_checkpoint(epoch, net, optimizer, train_loss, train_accuracy, val_acc, test_accuracy, checkpoint_path):
    """
    Saves the final checkpoint of the model, optimizer, and training metrics.

    Args:
        epoch (int): The final epoch number.
        net (torch.nn.Module): The neural network model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        train_loss (float): The training loss at the final epoch.
        train_accuracy (float): The training accuracy at the final epoch.
        val_acc (float): The validation accuracy at the final epoch.
        test_accuracy (float): The test accuracy at the final epoch.
        checkpoint_path (str): The file path where the final checkpoint will be saved.

    Returns:
        None
    """
    
    final_checkpoint = {
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'train_acc': train_accuracy,
    'val_acc': val_acc,
    'test_acc': test_accuracy
    }
    torch.save(final_checkpoint, checkpoint_path)
    print(f"Final model checkpoint saved")
    

def load_model(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model