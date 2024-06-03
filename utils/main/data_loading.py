import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset


def get_data_loaders(data_dir, batch_size=32,
                     resize=(256, 256), crop=(224, 224),
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                     pytorch_dataset = False):
    '''
    Returns data loaders for training, validation, and testing datasets.

    Parameters:
    - data_dir (str): The directory path where the data is stored.
    - batch_size (int): The batch size for the data loaders. Default is 32.
    - resize (tuple): The size to which the images will be resized. Default is (256, 256).
    - crop (tuple): The size of the cropped images. Default is (224, 224).
    - mean (list): The mean values for image normalization. Default is [0.485, 0.456, 0.406].
    - std (list): The standard deviation values for image normalization. Default is [0.229, 0.224, 0.225].

    Returns:
    - train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
    - val_loader (torch.utils.data.DataLoader): The data loader for the validation dataset.
    - test_loader (torch.utils.data.DataLoader): The data loader for the testing dataset.
    '''
    if not pytorch_dataset:
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        test_dir = os.path.join(data_dir, 'test')

    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    if pytorch_dataset:
        if data_dir == 'pets':
            dataset = datasets.OxfordIIITPet(root='data', split='trainval', download=True)
            data_len = len(dataset)
            train_indices, val_indices = random_split(range(data_len), [data_len*0.7, data_len*0.3])
            train_dataset = Subset(
            datasets.OxfordIIITPet(root='data', split='trainval', transform=train_transform, download=True),
                train_indices
            )
            val_dataset = Subset(
                datasets.OxfordIIITPet(root='data', split='trainval', transform=val_transform, download=True),
                val_indices
            )
            test_dataset = datasets.OxfordIIITPet(root='data', split='test', transform=test_transform, download=True)
        if data_dir == 'aircraft':
            train_dataset = datasets.FGVC_Aircraft(root='data', split='train', transform=train_transform, download=True)
            val_dataset = datasets.FGVC_Aircraft(root='data', split='val', transform=val_transform, download=True)
            test_dataset = datasets.FGVC_Aircraft(root='data', split='test', transform=test_transform, download=True)
    else:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader


def get_data_loaders_comp(data_dir, batch_size=32,
                     resize=(256, 256), crop=(224, 224),
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Returns the data loaders for training and validation datasets.

    Args:
        data_dir (str): The directory path where the data is stored.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 32.
        resize (tuple, optional): The size to which the images will be resized. Defaults to (256, 256).
        crop (tuple, optional): The size of the cropped images. Defaults to (224, 224).
        mean (list, optional): The mean values for normalization. Defaults to [0.485, 0.456, 0.406].
        std (list, optional): The standard deviation values for normalization. Defaults to [0.229, 0.224, 0.225].

    Returns:
        tuple: A tuple containing the training data loader and validation data loader.
    '''
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader


def get_test_loader_comp(data_dir, batch_size=32,
                     resize=(256, 256), crop=(224, 224),
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    test_dir = os.path.join(data_dir, 'test')

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_dataset = datasets.DatasetFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return test_loader


def get_label_ids(train_dir):
        label_ids = []
        classes = sorted(os.listdir(train_dir))
        for class_name in classes:
            if os.path.isdir(os.path.join(train_dir, class_name)):
                class_id = class_name.split('_')[0]
                label_ids.append(class_id)
        return label_ids
