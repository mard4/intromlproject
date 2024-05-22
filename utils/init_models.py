import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
import timm
    
def init_model(model_name, num_classes):
    """
    Given a model_name configured before in the main.py file, this function 
    initializes the model with the specified number of classes or freezing settings.

    Args:
        model_name (str): The name of the model to initialize.
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The initialized model.
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
        #add here new models
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
    and replace the classifier with a new one for fine-tuning
    """
    alexnet = torchvision.models.alexnet(pretrained=True)

    # Get the number of neurons in the second last layer
    in_features = alexnet.classifier[6].in_features

    # Re-initalize the output layer
    alexnet.classifier[6] = nn.Linear(in_features=in_features, out_features=num_classes)

    return alexnet

def initialize_efficientnet(num_classes):
    """Load the pre-trained EfficientNet-B7 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning
    """
    model_name = 'efficientnet-b7'

    # Load pre-trained EfficientNet model with ImageNet weights
    model = EfficientNet.from_pretrained(model_name, advprop=True)

    # Replace the classifier (fully connected layer) with a new one for fine-tuning
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, num_classes)
    return model

def initialize_adv_inception_v3(num_classes):
    """Load the pre-trained Inception V3 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning
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
    """Load the pre-trained Inception V3 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning
    FREEZED MODEL
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
        nn.Linear(2048, 1024),  # More features can be tuned based on your specific needs
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)  # Final output matching the number of classes
        )
    if hasattr(model, 'fc'):
        model.fc = new_classifier
    elif hasattr(model, 'classifier'):
        model.classifier = new_classifier
    else:
        raise AttributeError("Model doesn't have a recognizable final classifier layer.")
    return model

def initialize_adv_inception_v4(num_classes):
    """Load the pre-trained Inception V4 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning
    """
    model = timm.create_model('inception_v4', pretrained=True)
    model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
    return model

def initialize_adv_inception_v4_freeze(num_classes):
    """Load the pre-trained Inception V4 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning
    FREEZED MODEL
    """
    model = timm.create_model('inception_v4', pretrained=True)

    # Freeze all layers in the model
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.last_linear.in_features  # Get the number of input features of the last layer

    # Define a new classifier
    class CustomClassifier(nn.Module):
        def init(self, num_features, num_classes):
            super(CustomClassifier, self).init()
            self.new_layers = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        def forward(self, x):
            return self.new_layers(x)

    # Replace the original classifier with the new classifier
    model.last_linear = CustomClassifier(num_features, num_classes)
    return model

def initialize_densenet201(num_classes):
    """Load the pre-trained DenseNet-201 model with ImageNet weights
    and replace the classifier with a new one for fine-tuning
    """
    model = torchvision.models.densenet201(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model