import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.init_models import *
from utils.init_checkpoints import *
from utils.trainloop import *
from utils.read_dataset import *
from utils.optimizer import *

def main(run_name,
        batch_size,
        device,
        checkpoint_path,
        learning_rate,
        weight_decay,
        momentum,
        betas,
        epochs,
        criterion,
        optimizer_type,
        optim,
        param_groups,
        visualization_name,
        img_root,
        save_every,
        init_model,
        transform,
        val_loader):
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('training_logger')
    
    writer = SummaryWriter(log_dir=f"runs/{run_name}")  # TensorBoard
       
    train_dataset, val_dataset, test_dataset = read_dataset(img_root, transform)
    trainloader, valloader, testloader = get_data_loader(train_dataset, val_dataset, test_dataset, batch_size)
    plot_firstimg(train_dataset)

    net = init_model.to(device)
    
    logger.info("Starting to train: %s", visualization_name)
    
    optimizer = get_optimizer(
                            optimizer_type=optimizer_type,
                            model=net, 
                            lr=learning_rate, 
                            wd=weight_decay, 
                            momentum=momentum, 
                            betas=betas,
                            param_groups=param_groups, 
                            optim=optim
                        )
    logger.info("Optimizer: %s", optimizer)
   
    pbar = tqdm(range(epochs), desc="Training", position=0, leave=True)
    for e in pbar:
        train_loss, train_accuracy, val_loss, val_accuracy = train_one_epoch(net, trainloader, valloader, optimizer, criterion, device)

        test_loss, test_accuracy = test(net, testloader, criterion, device)
        
        checkpoint_perepoch(e, save_every, net, optimizer, train_loss, train_accuracy, val_accuracy, checkpoint_path)

        writer.add_scalar("train/loss", train_loss, e + 1)
        writer.add_scalar("train/accuracy", train_accuracy, e + 1)
        writer.add_scalar("val/loss", val_loss, e + 1)
        writer.add_scalar("val/accuracy", val_accuracy, e + 1)
        writer.add_scalar("test/loss", test_loss, e + 1)
        writer.add_scalar("test/accuracy", test_accuracy, e + 1)

        pbar.set_postfix(train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)
        pbar.update(1)

        
    logger.info("After training:")
    train_loss, train_accuracy = test(net, trainloader, criterion, device)
    test_loss, test_accuracy = test(net, testloader, criterion, device)

    logger.info(f"Training loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    logger.info(f"Validation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
    logger.info(f"Test loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    
    final_checkpoint(epochs, net, optimizer, train_loss, train_accuracy, val_accuracy, checkpoint_path)
    
    writer.add_scalar("train/loss", train_loss, epochs + 1)
    writer.add_scalar("train/accuracy", train_accuracy, epochs + 1)
    writer.add_scalar("test/loss", test_loss, epochs + 1)
    writer.add_scalar("test/accuracy", test_accuracy, epochs + 1)

    writer.close()

    return net, optimizer
