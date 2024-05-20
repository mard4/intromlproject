from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers

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
        transform):
    
    writer = SummaryWriter(log_dir=f"runs/{run_name}")  #tensorboad
       
    train_dataset, val_dataset, test_dataset = read_dataset(img_root, transform)
    trainloader, valloader, testloader = get_data_loader(train_dataset, val_dataset, test_dataset, batch_size)
    plot_firstimg(train_dataset)

    # Instantiates the model
    #num_classes = len(train_dataset.classes)
    net = init_model.to(device)
    
    print("Starting to train: ", visualization_name)
    
    # cost function
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
    print("Optimizer: ", optimizer)
   
    # Range over the number of epochs
    pbar = tqdm(range(epochs), desc="Training", position=0, leave=True)
    for e in range(epochs):
        # Train and log
        train_loss, train_accuracy, val_loss, val_accuracy = train_one_epoch(net, trainloader,valloader, optimizer, criterion)

        test_loss, test_accuracy = test(net, testloader, criterion)
        
        # checkpoint
        checkpoint_perepoch(e, save_every, net, optimizer, train_loss, train_accuracy, val_accuracy, checkpoint_path)   # correct with validation accyraacy

        # Add values to plots  (tensorboard)
        writer.add_scalar("train/loss", train_loss, e + 1)
        writer.add_scalar("train/accuracy", train_accuracy, e + 1)
        writer.add_scalar("val/loss", val_loss, e + 1)
        writer.add_scalar("val/accuracy", val_accuracy, e + 1)
        
        writer.add_scalar("test/loss", test_loss, e + 1)
        writer.add_scalar("test/accuracy", test_accuracy, e + 1)

        pbar.set_postfix(train_loss=train_loss, train_accuracy=train_accuracy, test_loss=test_loss, test_accuracy=test_accuracy)
        pbar.update(1)

    # Compute and print final metrics
    print("After training:")
    train_loss, train_accuracy = test(net, trainloader, criterion)
    test_loss, test_accuracy = test(net, testloader, criterion)

    print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
    print(f"\tValidation loss {val_loss:.5f}, Validaition accuracy {val_loss:.2f}")
    print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")
    
    final_checkpoint(e, net, optimizer, train_loss, train_accuracy, test_accuracy, test_accuracy, checkpoint_path)   # correct with validation accyraacy
    
    # Add values to plots
    writer.add_scalar("train/loss", train_loss, epochs + 1)
    writer.add_scalar("train/accuracy", train_accuracy, epochs + 1)
    writer.add_scalar("test/loss", test_loss, epochs + 1)
    writer.add_scalar("test/accuracy", test_accuracy, epochs + 1)

    # Close the logger
    writer.close()

    return net, optimizer

