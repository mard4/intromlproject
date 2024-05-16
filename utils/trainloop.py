import torch
from tqdm import tqdm

###################################################################### test with this
## reminder: add the validation set calculation
    
def test(net, data_loader, cost_function, device="cuda"):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # Set the network to evaluation mode
    net.eval()

    # Disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():
        # Iterate over the test set
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Load data into GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = net(inputs)

            # Loss computation
            loss = cost_function(outputs, targets)

            # Fetch prediction and loss value
            samples+=inputs.shape[0]
            cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
            _, predicted = outputs.max(1)

            # Compute accuracy
            cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples, cumulative_accuracy / samples * 100


def train_one_epoch(net, train_loader, val_loader, optimizer, cost_function, device="cuda"):
    """with validation"""
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # Set the network to training mode: particularly important when using dropout!
    net.train()

    # Iterate over the training set
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Load data into GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = net(inputs)

        # Loss computation
        loss = cost_function(outputs, targets)

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

        # Gradients reset
        optimizer.zero_grad()

        # Fetch prediction and loss value
        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        cumulative_accuracy += predicted.eq(targets).sum().item()

    train_loss = cumulative_loss / samples
    train_accuracy = cumulative_accuracy / samples * 100

    # Validation step
    val_loss, val_accuracy = validate(net, val_loader, cost_function, device)

    return train_loss, train_accuracy, val_loss, val_accuracy

def validate(net, val_loader, cost_function, device="cuda"):
    net.eval()  # Set the network to evaluation mode

    val_samples = 0.0
    val_cumulative_loss = 0.0
    val_cumulative_accuracy = 0.0

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net(inputs)
            loss = cost_function(outputs, targets)

            val_samples += inputs.shape[0]
            val_cumulative_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            val_cumulative_accuracy += predicted.eq(targets).sum().item()

    val_loss = val_cumulative_loss / val_samples
    val_accuracy = val_cumulative_accuracy / val_samples * 100

    return val_loss, val_accuracy



# def train_one_epoch(net, data_loader, optimizer, cost_function, device="cuda"):
#     """without validation"""
#     samples = 0.0
#     cumulative_loss = 0.0
#     cumulative_accuracy = 0.0

#     # Set the network to training mode: particularly important when using dropout!
#     net.train()

#     # Iterate over the training set
#     for batch_idx, (inputs, targets) in enumerate(data_loader):
#         # Load data into GPU
#         inputs = inputs.to(device)
#         targets = targets.to(device)

#         # Forward pass
#         outputs = net(inputs)

#         # Loss computation
#         loss = cost_function(outputs,targets)

#         # Backward pass
#         loss.backward()

#         # Parameters update
#         optimizer.step()

#         # Gradients reset
#         optimizer.zero_grad()

#         # Fetch prediction and loss value
#         samples += inputs.shape[0]
#         cumulative_loss += loss.item()
#         _, predicted = outputs.max(dim=1)

#     # Compute training accuracy
#     cumulative_accuracy += predicted.eq(targets).sum().item()

#     return cumulative_loss / samples, cumulative_accuracy / samples * 100


#def train(num_epochs, device, model, optimizer, criterion, train_loader, val_loader, test_loader, checkpoint_path, save_every=1, freeze=False):

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#         train_loss = running_loss / len(train_loader)
#         train_acc = 100. * correct / total

#         # Validation
#         model.eval()
#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()

#         val_acc = 100. * correct / total
#         print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Validation Acc: {val_acc:.2f}%')

#         # Save checkpoint every 'save_every' epochs
#         if (epoch + 1) % save_every == 0:
#             checkpoint = {
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'train_loss': train_loss,
#                 'train_acc': train_acc,
#                 'val_acc': val_acc
#             }
#             torch.save(checkpoint, checkpoint_path)
#             print(f"Checkpoint saved at epoch {epoch + 1}")

#     # Evaluate on test set and save final model checkpoint
#     test_accuracy = evaluate_model(model, test_loader, device)
#     print(f'Finished Training. Test Accuracy: {test_accuracy:.2f}%')

#     final_checkpoint = {
#         'epoch': num_epochs,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'train_loss': train_loss,
#         'train_acc': train_acc,
#         'val_acc': val_acc,
#         'test_acc': test_accuracy
#     }
#     torch.save(final_checkpoint, checkpoint_path)
#     print(f"Final model checkpoint saved")
#     return

    
# def evaluate_model(model, test_loader, device):
#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader, desc="Evaluating (Test)"):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#     test_accuracy = 100. * correct / total
#     return test_accuracy
