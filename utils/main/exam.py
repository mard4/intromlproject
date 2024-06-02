from http.client import responses
import requests
import json
import torch


def test_model(model, test_loader, label_ids, device='cuda'):
    '''
    This function evaluates a trained model on the test dataset and logs the results.
    '''
    model.eval()

    # Initialize an empty dictionary to store predictions
    preds = {}

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, images_names in test_loader:
            # Move images to the same device as the model
            images = images.to(next(model.parameters()).device)
            
            # Get model predictions
            outputs = model(images)
            
            # Get the predicted labels
            _, predicted = torch.max(outputs, 1)
            
            # Store the predictions in the dictionary
            for i, pred in enumerate(predicted):
                preds[images_names[i]] = label_ids[pred.item()]




def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


