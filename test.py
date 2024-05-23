import torch
import os
import logging
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.read_dataset import *
from utils.init_models import *
from utils.trainloop import * 
from utils.init_checkpoints import *
 

def main(model_name, checkpoint_path, img_root, batch_size, device, transform):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('testing_logger')
    

    # Prepare the test dataset and dataloader
    train_data, val_data, test_data = read_dataset(img_root, transform=transform)
    train_loader, val_loader, test_loader = get_data_loader(train_data, val_data, test_data, batch_size)

    num_classes = len(train_data.classes)
    model = init_model(model_name, num_classes).to(device)
    # Load the model
    print("model_loaded")
    model = load_model(checkpoint_path, model, device)
    model.eval()


    # Define the criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Run the test loop
    test_loss, test_accuracy = test(model, test_loader, criterion, device)

    logger.info(f"Test loss: {test_loss:.5f}, Test accuracy: {test_accuracy:.2f}")

if __name__ == "__main__":
    # Replace these paths and parameters as needed
    model_name = "densenet201"
    checkpoint_path = "/home/disi/ml/datasetmodels/checkpoint_densenet201.pth/checkpoint_epoch_0_20240523_095440.pth"
    img_root = "/home/disi/ml/dataset/Aerei"
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the same transform as used during training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    main(model_name, checkpoint_path, img_root, batch_size, device, transform)
