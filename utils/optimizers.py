import torch.optim as optim

def create_custom_optimizer(model_parameters, config):
    """
    Creates a custom optimizer with specified parameters.

    Args:
        model_parameters: Parameters of the model to be optimized.
        config (dict): Configuration dictionary containing optimizer settings.

    Returns:
        optimizer: Custom optimizer instance.
    """
    optimizer_type = config.get('type', 'Adam')
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0)
    momentum = config.get('momentum', 0)

    if optimizer_type == 'Adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        return optim.SGD(model_parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
