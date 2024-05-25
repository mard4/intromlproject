import torch
import logging
import wandb
from http.client import responses
import requests
import json

"""
Image Sizes:
    - AlexNet: 224x224
    - DenseNet: 224x224
    - Inception: 299x299
    - ResNet: 224x224
    - VGG: 224x224

Mean, Std, number of classes for Datasets:
    - CUB-200-2011: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=200
    - Stanford Dogs: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=120
    - FGVC Aircraft: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=102
    - Flowers102: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], classes=102
"""

def test_model_exam(net, test_loader, cost_function, device="cuda"):
    """
    Test the model on the test dataset, log the results, and submit them to a competition server.

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
    preds = {}

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = cost_function(outputs, targets)

            samples += inputs.size(0)
            cumulative_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()
            
            # Collect predictions for submission
            for idx, pred in enumerate(predicted):
                preds[str(i * test_loader.batch_size + idx)] = pred.item()

    test_loss = cumulative_loss / samples
    test_accuracy = cumulative_accuracy / samples * 100

    logger.info(f"Test: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    # Log metrics to wandb
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    })

    # Prepare and submit the results
    res = {
        "images": preds,
        "groupname": "your_group_name"
    }
    print(res)
    submit(res)

    return test_loss, test_accuracy

def submit(results, url="https://competition-production.up.railway.app/results/"):
    """
    Submits the results to the specified competition results endpoint.

    Args:
        results (dict): Dictionary containing the predictions and group name.
        url (str): URL to which the results are submitted.
    """
    res = json.dumps(results)
    response = requests.post(url, data=res, headers={"Content-Type": "application/json"})
    try:
        result = json.loads(response.text)
        print(f"Accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
