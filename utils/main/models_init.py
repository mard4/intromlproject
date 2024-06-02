import torch.nn as nn
import torchvision
import torch
from efficientnet_pytorch import EfficientNet
import timm

def init_model(model_name, num_classes, dropout_rate):
    """
    Given a model_name configured before in the main.py file, this function 
    initializes the model with the specified number of classes or freezing settings.

    Args:
        model_name (str): The name of the model to initialize.
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The initialized model.

    Raises:
        ValueError: If the model_name is not recognized.
    """
    # Dictionary mapping model names to their initialization functions
    model_initializers = {
        "densenet201": initialize_densenet201,
        'efficientnetv2': initialize_efficientnetv2,
        "seresnet50": initialize_SENet,
        "vit": initialize_ViT
    }
    
    # Get the initializer function based on the model_name
    if model_name in model_initializers:
        model = model_initializers[model_name](num_classes, dropout_rate)
    else:
        raise ValueError(f"Model name '{model_name}' not found")
    
    return model

def load_checkpoint(model, checkpoint_path, device="cuda"):
    """
    Load model weights from a checkpoint.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        checkpoint_path (str): Path to the checkpoint file.
        device (str): The device to map the checkpoint (default is "cuda").

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model

def initialize_densenet201(num_classes,dropout_rate = 0.5):
    """
    Load the pre-trained DenseNet-201 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning.

    Args:
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The DenseNet-201 model with the modified classifier.
    """
    model = torchvision.models.densenet201(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.dropout = nn.Dropout(dropout_rate)
    return model

def initialize_efficientnetv2(num_classes, dropout_rate = 0):
    """
    Load the pre-trained EfficientNetV2S model with ImageNet weights
    and replace the classifier with a new one for fine-tuning.
    The first layer is frozen.

    Args:
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The EfficientNetV2S  model with the modified classifier.
    """
    
    model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    
    # Freeze the first layer
    first_layer = model.features[0]
    for param in first_layer.parameters():
        param.requires_grad = False
    
    # Modify the classifier for the desired number of output classes
    in_features = model.classifier[1].in_features  # Access the in_features of the second layer of the classifier
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.dropout = nn.Dropout(dropout_rate)

    return model

def initialize_SENet(num_classes, dropout_rate):
    from utils.main.models.senet import CustomSqueezeNet
    model = CustomSqueezeNet(num_classes, dropout_rate)
    return model

def initialize_ViT(num_classes, dropout_rate):
    from utils.main.models.vit import ViTFineTuner
    model = ViTFineTuner(num_classes, dropout_rate)
    return model