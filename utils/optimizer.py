import torch

def get_optimizer(optimizer_type, model, lr, wd, momentum=0.9, betas=(0.9, 0.999), param_groups=None, optim=torch.optim.SGD):
    """Get the optimizer based on the type."""
    if optimizer_type == "Fixed":
        return simple_optimizer(model, lr, wd, momentum, betas, optim)
    elif optimizer_type == "Custom":
        if param_groups is None:
            raise ValueError("param_groups must be provided for custom optimizer")
        return custom_optimizer(model, lr, wd, param_groups,optim, momentum=momentum, betas=betas)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")



def simple_optimizer(model, lr, wd, momentum=0.9, betas=(0.9, 0.999), optim=torch.optim.SGD):
    """Simple optimizer with the same learning rate for all parameters."""
    if optim == torch.optim.SGD:
        return optim(params=model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    elif optim == torch.optim.Adam or torch.optim.AdamW:
        return torch.optim.Adam(params=model.parameters() ,lr=lr, weight_decay=wd, betas=betas)
    else:
        raise ValueError(f"Unsupported optimizer: {optim}")
    

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


# Example usage:
# model = ...  # Your model here
# param_groups = [
#     {'prefixes': ['classifier'], 'lr': 0.001},
#     {'prefixes': ['features']}
# ]
# optimizer = get_optimizer('Custom', model, lr=0.01, wd=0.0001, momentum=0.9, betas=(0.9, 0.999), param_groups=param_groups)


# Example usage:
# model = ...  # Your model here
# param_groups = [
#     {'prefixes': ['classifier'], 'lr': 0.001},
#     {'prefixes': ['features']}
# ]
# optimizer = custom_optimizer(model, lr=0.01, wd=0.0001, param_groups=param_groups, optim_type='Adam')


# Example usage:
# model = ...  # Your model here
# optimizer_type = "Custom"
# lr = 0.01
# wd = 0.0005
# momentum = 0.9
# param_groups = [
#     {'prefixes': ['classifier'], 'lr': lr * 10},
#     {'prefixes': ['features']}
# ]
# optimizer = get_optimizer(optimizer_type, model, lr, wd, momentum, param_groups)

# def optimizer_alexnet(model, lr, wd, momentum):
#     """Alexnet optimizer with different learning rates for the final layer"""
#     # We will create two groups of weights, one for the newly initialized layer
#     # and the other for rest of the layers of the network
#     final_layer_weights = []
#     rest_of_the_net_weights = []

#     # Iterate through the layers of the network
#     for name, param in model.named_parameters():
#         if name.startswith('classifier.6'):
#             final_layer_weights.append(param)
#         else:
#             rest_of_the_net_weights.append(param)

#     # Assign the distinct learning rates to each group of parameters
#     optimizer = torch.optim.SGD([
#         {'params': rest_of_the_net_weights},
#         {'params': final_layer_weights, 'lr': lr}
#     ], lr=lr / 10, weight_decay=wd, momentum=momentum)
    
#     return optimizer
