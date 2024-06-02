import torch.optim as optim
import torch

def init_optimizer(model, config):
    """
    Initializes and returns an optimizer based on the provided configuration.

    Args:
        model (torch.nn.Module): The model for which the optimizer is initialized.
        config (dict): A dictionary containing the configuration parameters for the optimizer.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Raises:
        ValueError: If the provided optimizer type is not supported.

    """
    if config['optimizer_type'] == 'simple':
        optimizer = simple_optimizer(model, config)
    elif config['optimizer_type'] == 'custom':
        optimizer = custom_optimizer(model, config)
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer_type']}, choose between simple and custom")
    print("Optimizer", optimizer)
    return optimizer
        
def simple_optimizer(model, config):
    config['optimizer'] = config['optimizer'].lower()
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=config['momentum'])
    elif config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer']}")
    return optimizer

def custom_optimizer(model, config):
    """
    Create an optimizer with different learning rates for different parts of the model.

    Args:
        model (torch.nn.Module): The model with layers to be fine-tuned.
        lr (float): The learning rate for the optimizer.
        wd (float): The weight decay (L2 penalty) for the optimizer.
        momentum (float): The momentum for the optimizer.

    Returns:
        torch.optim.Optimizer: The configured optimizer.
    """
    final_layer_weights = []
    rest_of_the_net_weights = []

    # Iterate through the layers of the network
    for name, param in model.named_parameters():
        if 'classifier' in name:
            final_layer_weights.append(param)
        else:
            rest_of_the_net_weights.append(param)

    # Assign the distinct learning rates to each group of parameters
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': rest_of_the_net_weights},
            {'params': final_layer_weights, 'lr': config['learning_rate']}
        ], lr=config['learning_rate'] / 10, weight_decay=config['weight_decay'], momentum=config['momentum'])
        
    if config['optimizer'] == 'Adam' or 'AdamW':
        optimizer = torch.optim.Adam([
            {'params': rest_of_the_net_weights},
            {'params': final_layer_weights, 'lr': config['learning_rate']}
        ], lr=config['learning_rate'] / 10, weight_decay=config['weight_decay'])

    return optimizer
