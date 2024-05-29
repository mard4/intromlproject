import torch.nn as nn
import torchvision
import torch    
from efficientnet_pytorch import EfficientNet
import timm
from utils.custom_models import *

def init_model(model_name, num_classes):
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
        "inceptionv4": initialize_adv_inception_v4,
        "inceptionv4_freeze": initialize_adv_inception_v4_freeze,
        "inceptionv3": initialize_adv_inception_v3,
        "inceptionv3_freeze": initialize_adv_inception_v3_freeze,
        "efficientnet": initialize_efficientnet,
        "alexnet": initialize_alexnet,
        "densenet201": initialize_densenet201,
        'efficientnetv2': initialize_efficientnetv2,
        'vit_base_patch16_224': initialize_vit_base_patch16_224,
        "efficientnetv2_freeze": initialize_efficientnetv2_freeze,
        "seresnet50": initialize_SENet,
        "initialize_densenet201_freeze_1st" :initialize_densenet201_freeze_1st_block,
        "seresnet50_freeze": initialize_SENet_freeze_except_last,
        "vit": initialize_ViT,
        "vit_freeze": initialize_ViT_freeze_except_last
        # add here new models
    }
    
    # Get the initializer function based on the model_name
    if model_name in model_initializers:
        model = model_initializers[model_name](num_classes)
    else:
        raise ValueError(f"Model name '{model_name}' not found")
    
    return model

def initialize_alexnet(num_classes):
    """
    Load the pre-trained Alexnet model with ImageNet weights
    and replace the classifier with a new one for fine-tuning.

    Args:
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The Alexnet model with the modified classifier.
    """
    alexnet = torchvision.models.alexnet(pretrained=True)

    # Get the number of neurons in the second last layer
    in_features = alexnet.classifier[6].in_features

    # Re-initialize the output layer
    alexnet.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes)

    return alexnet

def initialize_efficientnet(num_classes):
    """
    Load the pre-trained EfficientNet-B7 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning.

    Args:
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The EfficientNet-B7 model with the modified classifier.
    """
    model_name = 'efficientnet-b7'

    # Load pre-trained EfficientNet model with ImageNet weights
    model = EfficientNet.from_pretrained(model_name, advprop=True)

    # Replace the classifier (fully connected layer) with a new one for fine-tuning
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, num_classes)
    return model

def initialize_adv_inception_v3(num_classes):
    """
    Load the pre-trained Inception V3 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning.

    Args:
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The Inception V3 model with the modified classifier.

    Raises:
        AttributeError: If the model doesn't have a recognizable final classifier layer.
    """
    model = timm.create_model('adv_inception_v3', pretrained=True)
    if hasattr(model, 'fc'):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif hasattr(model, 'classifier'):
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    else:
        raise AttributeError("Model doesn't have a recognizable final classifier layer.")
    return model

def initialize_adv_inception_v3_freeze(num_classes):
    """
    Load the pre-trained Inception V3 model with ImageNet weights,
    replace the classifier with a new one for fine-tuning, and freeze the base layers.

    Args:
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The Inception V3 model with the modified classifier and frozen base layers.

    Raises:
        AttributeError: If the model doesn't have a recognizable final classifier layer.
    """
    model = timm.create_model('adv_inception_v3', pretrained=True)
    # Freeze all layers in the model first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the later layers
    if hasattr(model, 'features'):
        # Unfreeze the last few layers
        for param in model.features[-2:].parameters():
            param.requires_grad = True

    new_classifier = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    if hasattr(model, 'fc'):
        model.fc = new_classifier
    elif hasattr(model, 'classifier'):
        model.classifier = new_classifier
    else:
        raise AttributeError("Model doesn't have a recognizable final classifier layer.")
    return model

def initialize_adv_inception_v4(num_classes):
    """
    Load the pre-trained Inception V4 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning.

    Args:
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The Inception V4 model with the modified classifier.
    """
    model = timm.create_model('inception_v4', pretrained=True)
    model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
    return model

def initialize_adv_inception_v4_freeze(num_classes):
    """
    Load the pre-trained Inception V4 model with ImageNet weights,
    replace the classifier with a new one for fine-tuning, and freeze the base layers.

    Args:
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The Inception V4 model with the modified classifier and frozen base layers.
    """
    model = timm.create_model('inception_v4', pretrained=True)

    # Freeze all layers in the model
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    class CustomClassifier(nn.Module):
        def __init__(self, num_features, num_classes):
            super(CustomClassifier, self).__init__()
            self.new_layers = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        def forward(self, x):
            return self.new_layers(x)

    num_features = model.last_linear.in_features
    model.last_linear = CustomClassifier(num_features, num_classes)
    return model

def initialize_densenet201(num_classes):
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
    return model


import torchvision.models as models
import torch.nn as nn

def initialize_densenet201_freeze_1st_block(num_classes, freeze_first_n_blocks=1):
    # Load the pre-trained DenseNet-201 model
    model = models.densenet201(pretrained=True)
    
    # Accessing features of the DenseNet
    features = model.features
    
    # Freeze the specified number of dense blocks
    block_count = 0
    for child in features.children():
        if isinstance(child, nn.Sequential):  # Each dense block is a Sequential module
            block_count += 1
            if block_count <= freeze_first_n_blocks:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                break

    # Replacing the classifier with a new one for the given number of classes
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
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

def initialize_efficientnetv2(num_classes):
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

    return model

def initialize_vit_base_patch16_224(num_classes):
    """roba EnricoMardeen"""
    model_name = 'vit_base_patch16_224'
    model = timm.create_model(model_name, pretrained=True)
    freeze_layers_except_last = False
    if freeze_layers_except_last:

        # freezing
        for param in model.parameters():
            param.requires_grad = False

        # sfreezing
        model.head = nn.Linear(model.head.in_features, num_classes)
        for param in model.head.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')
    print(f'Frozen parameters: {frozen_params}')


    # if freeze_layers_except_last:
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    # else:
    #     optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model
    # model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # return model

def initialize_efficientnetv2_freeze(num_classes):
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
    
    # Freeze all layers in the model first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the later layers
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    new_classifier = nn.Sequential(
        nn.Linear(1280, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    model.classifier = new_classifier
    
    return model


def initialize_SENet(num_classes):
    """
    Load the pre-trained SEResNet50 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning.

    Args:
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The SEResNet50 model with the modified classifier.
    """
    model = SEResNet50(num_classes)
    # model = ResNet50(ResidualBlock, [3, 4, 6, 3], num_classes)
    return model

def initialize_SENet_freeze_except_last(num_classes):
    """
    Load the pre-trained SEResNet50 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning.
    The first layer is frozen.

    Args:
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The SEResNet50 model with the modified classifier.
    """
    model = SEResNet50(num_classes, freeze_layers_except_last = True)
    
    return model

def initialize_ViT(num_classes):
    '''
    Load the pre-trained ViT model with ImageNet weights
    '''
    model = ViT(num_classes)
    return model

def initialize_ViT_freeze_except_last(num_classes):
    '''
    Load the pre-trained ViT model with ImageNet weights
    '''
    model = ViT(num_classes, freeze_layers_except_last = True)
    return model
