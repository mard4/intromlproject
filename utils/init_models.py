import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
import timm
    
def init_model(model_name, num_classes):
    """
    Given a model_name configured before in file main.py, this function 
    initializes the model with the specified number of classes or freezing things.

    Args:
        model_name (_type_): _description_
        num_classes (_type_): _description_

    Returns:
        model
    """
    
    if model_name=="inceptionv4":
        model = initialize_adv_inception_v4(num_classes)
    elif model_name=="inceptionv4_freeze":
        model = initialize_adv_inception_v4_freeze(num_classes)
    elif model_name=="inceptionv3":
        model = initialize_adv_inception_v3(num_classes)
    elif model_name=="inceptionv3_freeze":
        model = initialize_adv_inception_v3_freeze(num_classes)
    elif model_name=="efficientnet":
        model = initialize_efficientnet(num_classes)
    elif model_name=="alexnet":
        model = initialize_alexnet(num_classes)
    else:
        raise ValueError("Model name not found")
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
    # You need to determine how many layers you want to unfreeze.
    # For Inception models, you might start by unfreezing the last few inception blocks
    # Here's an example assuming the model has a features block containing the layers
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


