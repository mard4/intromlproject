from torch.optim.lr_scheduler import StepLR

def init_scheduler(optimizer, config):
    """
    Initializes and returns a scheduler based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the scheduler.

    Returns:
        torch.optim.lr_scheduler: The initialized scheduler.

    Raises:
        ValueError: If the provided scheduler type is not supported.

    """
    if config['scheduler'] == True:
        scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    else:
        raise ValueError(f"Unsupported scheduler type: {config['scheduler_type']}, choose between step and plateau")
    print("Scheduler", scheduler)
    return scheduler