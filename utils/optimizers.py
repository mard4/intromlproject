import torch.optim as optim
import torch

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


def custom_optimizer(model, lr, wd, param_groups, optim, momentum=0.9, betas=(0.9, 0.999)):
    """Custom optimizer with different learning rates for specified parameter groups.
    Supports SGD and Adam optimizers.
    
    Args:
        model: The model containing the parameters to optimize.
        lr: Base learning rate.
        wd: Weight decay.
        param_groups: List of parameter group dictionaries, each containing 'prefixes' and optional specific 'lr'.
        optim: Optimizer class (e.g., torch.optim.SGD, torch.optim.Adam).
        momentum: Momentum factor (for SGD).
        betas: Coefficients used for computing running averages of gradient and its square (for Adam).
    
    Returns:
        optimizer: The configured optimizer.
    """
    params = []

    for group in param_groups:
        group_params = {'params': []}
        group_params.update(group)

        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in group['prefixes']):
                group_params['params'].append(param)

        params.append(group_params)

    if optim == torch.optim.SGD:
        optimizer = optim(params, lr=lr, weight_decay=wd, momentum=momentum)
    elif optim == torch.optim.Adam:
        optimizer = optim(params, lr=lr, weight_decay=wd, betas=betas)
    else:
        raise ValueError(f"Unsupported optimizer class: {optim}")

    return optimizer