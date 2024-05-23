import torch

def simple_optimizer(model, lr, wd, momentum=0.9, betas=(0.9, 0.999), optim=torch.optim.SGD):
    """
    Simple optimizer with the same learning rate for all parameters.

    Args:
        model (torch.nn.Module): The model to optimize.
        lr (float): Learning rate.
        wd (float): Weight decay.
        momentum (float, optional): Momentum factor (for SGD). Default is 0.9.
        betas (tuple, optional): Coefficients used for computing running averages of gradient and its square (for Adam). Default is (0.9, 0.999).
        optim (torch.optim.Optimizer): Optimizer class (e.g., torch.optim.SGD, torch.optim.Adam).

    Returns:
        torch.optim.Optimizer: The configured optimizer.

    Raises:
        ValueError: If an unsupported optimizer is provided.
    """
    if optim == torch.optim.SGD:
        return optim(params=model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    elif optim in [torch.optim.Adam, torch.optim.AdamW]:
        return optim(params=model.parameters(), lr=lr, weight_decay=wd, betas=betas)
    else:
        raise ValueError(f"Unsupported optimizer: {optim}")

def custom_optimizer(model, lr, wd, param_groups, optim, momentum=0.9, betas=(0.9, 0.999)):
    """
    Custom optimizer with different learning rates for specified parameter groups.
    Supports SGD and Adam optimizers.

    Args:
        model (torch.nn.Module): The model containing the parameters to optimize.
        lr (float): Base learning rate.
        wd (float): Weight decay.
        param_groups (list): List of parameter group dictionaries, each containing 'prefixes' and optional specific 'lr'.
        optim (torch.optim.Optimizer): Optimizer class (e.g., torch.optim.SGD, torch.optim.Adam).
        momentum (float, optional): Momentum factor (for SGD). Default is 0.9.
        betas (tuple, optional): Coefficients used for computing running averages of gradient and its square (for Adam). Default is (0.9, 0.999).

    Returns:
        torch.optim.Optimizer: The configured optimizer.

    Raises:
        ValueError: If an unsupported optimizer class is provided.
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
    elif optim in [torch.optim.Adam, torch.optim.AdamW]:
        optimizer = optim(params, lr=lr, weight_decay=wd, betas=betas)
    else:
        raise ValueError(f"Unsupported optimizer class: {optim}")

    return optimizer


def get_optimizer(optimizer_type, model, lr, wd, momentum=0.9, betas=(0.9, 0.999), param_groups=None, optim=torch.optim.SGD):
    """
    Get the optimizer based on the specified type.

    Args:
        optimizer_type (str): The type of optimizer to use ("simple" or "custom").
        model (torch.nn.Module): The model to optimize.
        lr (float): Learning rate.
        wd (float): Weight decay.
        momentum (float, optional): Momentum for optimizers that use it. Default is 0.9.
        betas (tuple, optional): Betas for optimizers that use them. Default is (0.9, 0.999).
        param_groups (list, optional): Parameter groups for custom optimizer. Required for "custom" type.
        optim (torch.optim.Optimizer, optional): The optimizer class to use. Default is torch.optim.SGD.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Raises:
        ValueError: If an unknown optimizer type is provided or param_groups is not provided for "custom" type.
    """
    optimizer_initializers = {
        "simple": simple_optimizer,
        "custom": custom_optimizer
    }

    if optimizer_type not in optimizer_initializers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    if optimizer_type == "custom":
        if param_groups is None:
            raise ValueError("param_groups must be provided for custom optimizer")
        return optimizer_initializers[optimizer_type](model, lr, wd, param_groups, optim, momentum, betas)
    else:
        return optimizer_initializers[optimizer_type](model, lr, wd, momentum, betas, optim)
