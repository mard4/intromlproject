import torch

# fare in modo che si salvi l epoca migliore
# elimina epoca precedente solo se trainval loss sono migliori

def checkpoint_perepoch(epoch, save_every, net, optimizer, train_loss, train_accuracy, val_acc, checkpoint_path):
    """
    Saves a checkpoint of the model, optimizer, and training metrics every specified number of epochs.

    Args:
        epoch (int): The current epoch number.
        save_every (int): The interval of epochs at which to save checkpoints.
        net (torch.nn.Module): The neural network model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        train_loss (float): The training loss at the current epoch.
        train_accuracy (float): The training accuracy at the current epoch.
        val_acc (float): The validation accuracy at the current epoch.
        checkpoint_path (str): The file path where the checkpoint will be saved.

    Returns:
        None
    """
    if (epoch + 1) % save_every == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_accuracy,
            'val_acc': val_acc
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")
    return

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
    