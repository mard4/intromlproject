import torch
import logging
import wandb

def test_model(net, test_loader, cost_function, device="cuda"):
    """
    Test the model on the test dataset and log the results.

    Args:
        net (torch.nn.Module): The model to test.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        cost_function (torch.nn.Module): The loss function.
        device (str): The device to run the testing on (default is "cuda").

    Returns:
        tuple: A tuple containing test loss and test accuracy.
    """
    logger = logging.getLogger('training_logger')
    net.eval()
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = cost_function(outputs, targets)

            samples += inputs.size(0)
            cumulative_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()

    test_loss = cumulative_loss / samples
    test_accuracy = cumulative_accuracy / samples * 100

    logger.info(f"Test: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")

    # Log metrics to wandb
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    })

    return test_loss, test_accuracy
