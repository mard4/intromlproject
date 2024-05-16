import torch

# fare in modo che si salvi l epoca migliore
# elimina epoca precedente solo se trainval loss sono migliori

def checkpoint_perepoch(e, save_every, net, optimizer, train_loss, train_accuracy, val_acc, checkpoint_path):
    """
    Save a checkpoint every save_every epochs

    Args:
        e (_type_): _description_
        save_every (_type_): _description_
        net (_type_): _description_
        optimizer (_type_): _description_
        train_loss (_type_): _description_
        train_accuracy (_type_): _description_
        val_acc (_type_): _description_
        checkpoint_path (_type_): _description_
    """
    if (e + 1) % save_every == 0:
        checkpoint = {
            'epoch': e + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_accuracy,
            'val_acc': val_acc
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {e + 1}")
    return

def final_checkpoint(e, net, optimizer, train_loss, train_accuracy, val_acc, test_accuracy, checkpoint_path):
    """
    Save the final checkpoint every save_every epochs

    Args:
        e (_type_): _description_
        save_every (_type_): _description_
        net (_type_): _description_
        optimizer (_type_): _description_
        train_loss (_type_): _description_
        train_accuracy (_type_): _description_
        val_acc (_type_): _description_
        checkpoint_path (_type_): _description_
    """
    
    final_checkpoint = {
    'epoch': e,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'train_acc': train_accuracy,
    'val_acc': val_acc,
    'test_acc': test_accuracy
    }
    torch.save(final_checkpoint, checkpoint_path)
    print(f"Final model checkpoint saved")
    