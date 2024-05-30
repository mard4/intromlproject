import torch.optim as optim
import torch


def create_optimizer(model, config):
    if config['optimizer_type'] == 'simple':
        optimizer = simple_optimizer(model, config)
    elif config['optimizer_type'] == 'custom':
        optimizer = custom_optimizer(model, config)
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer_type']}, choose between simple and custom")
    print("Optimizer", optimizer)
    return optimizer
        

    
def simple_optimizer(model, config):
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], momentum=config['momentum'])
    elif config['optimizer'] == 'Adam' or 'AdamW':
        #betas
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
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
# def custom_optimizer(model, config):
#     """Custom optimizer with different learning rates for specified parameter groups.
#     Supports SGD and Adam optimizers.
    
#     Args:
#         model: The model containing the parameters to optimize.
#         lr: Base learning rate.
#         wd: Weight decay.
#         param_groups: List of parameter group dictionaries, each containing 'prefixes' and optional specific 'lr'.
#         optim: Optimizer class (e.g., torch.optim.SGD, torch.optim.Adam).
#         momentum: Momentum factor (for SGD).
#         betas: Coefficients used for computing running averages of gradient and its square (for Adam).
    
#     Returns:
#         optimizer: The configured optimizer.
#     """
#     params = []
#     # Define parameter groups
#     # param_groups = [
#     #     {"params": model.layer1.parameters(), "lr": config['learning_rate_layer1']},
#     #     {"params": model.layer2.parameters(), "lr": config['learning_rate_layer2']},
#     #     # Add more parameter groups as needed
#     # ]
#     param_groups = {
#         "params": model.parameters(),
#         "lr": config['learning_rate'],
#         "weight_decay": config['weight_decay']
#     }
#     for group in param_groups:
#         group_params = {'params': []}
#         group_params.update(group)

#         for name, param in model.named_parameters():
#             if any(name.startswith(prefix) for prefix in group['prefixes']):
#                 group_params['params'].append(param)

#         params.append(group_params)

#     optimizer = simple_optimizer(model, config)

#     return optimizer