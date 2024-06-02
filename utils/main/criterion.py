import torch.nn as nn

def init_criterion(criterion_name):
    '''
    Initialize a criterion based on the given criterion name.

    Parameters:
        criterion_name (str): The name of the criterion to initialize.

    Returns:
        criterion: The initialized criterion.

    Raises:
        ValueError: If the given criterion name is not recognized.

    Right now there is just CrossEntropyLoss, but we can add more in the future.
    '''
    if criterion_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Criterion not found")
    
    return criterion