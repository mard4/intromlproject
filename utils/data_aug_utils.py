import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img
import PIL
import shutil
from sklearn.model_selection import train_test_split

def setup_data_generator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'):
    """
    Set up and return a Keras ImageDataGenerator with customizable augmentation parameters.

    Parameters:
    - rotation_range (int): Degree range for random rotations. Default is 40 degrees.
    - width_shift_range (float): Fraction of total width for random horizontal shifts. Default is 0.2.
    - height_shift_range (float): Fraction of total height for random vertical shifts. Default is 0.2.
    - shear_range (float): Shear intensity (shear angle in counter-clockwise direction in degrees). Default is 0.2.
    - zoom_range (float): Range for random zoom. Can be a single float or a tuple. Default is 0.2.
    - horizontal_flip (bool): Randomly flip inputs horizontally. Default is True.
    - fill_mode (str): Points outside the boundaries are filled according to the given mode: 
      'constant', 'nearest', 'reflect', or 'wrap'. Default is 'nearest'.

    Returns:
    - An instance of ImageDataGenerator configured with the specified augmentation parameters.
    """
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )

def load_data_from_directory(directory_path, target_size=(150, 150), batch_size=32):
    """
    Load data from a directory and prepare it for training using the configured ImageDataGenerator.

    Parameters:
    - directory_path (str): The path to the directory where the images are stored. This directory
      should be organized with subdirectories for each class, each containing the relevant images.
    - target_size (tuple): The dimensions to which all images found will be resized. It should be
      a tuple of two integers, (height, width). The default is (150, 150), which adjusts the size of
      the images to 150x150 pixels.
    - batch_size (int): The size of the batches of data (number of images). The default is 32,
      which means the generator will take thirty-two images at a time along with their labels.

    Returns:
    - A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch
      of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.

    Note:
    - The images within the directory_path are expected to be organized in subdirectories, each
      named after one of the classes. For instance, if you have two classes 'cats' and 'dogs', then
      your directory structure should be:
        /path/to/directory_path/cats/ <-- contains all cat images
        /path/to/directory_path/dogs/ <-- contains all dog images
    """
    datagen = setup_data_generator()  # Assumes setup_data_generator() is defined elsewhere
    return datagen.flow_from_directory(
        directory_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )





def save_augmented_images(images, labels, base_output_dir, class_indices):
    """
    Save augmented images to labeled directories within their respective class subfolders.

    Parameters:
    - images: Array of images to be saved.
    - labels: Corresponding labels for the images, used to determine subfolder.
    - base_output_dir (str): Base directory to store the class subdirectories.
    - class_indices (dict): Dictionary mapping class names to label indices.
    """
    # Reverse class indices to get label names from indices
    label_names = {v: k for k, v in class_indices.items()}
    for idx, (image, label) in enumerate(zip(images, labels)):
        # Get class label
        label_name = label_names[np.argmax(label)]  # label.argmax() may also be used, depending on your label format
        class_dir = os.path.join(base_output_dir, label_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        # Save image
        img = array_to_img(image)
        img.save(os.path.join(class_dir, f'aug_{idx}.png'))



# Assuming all images are in a single directory and categorized by folders
def create_train_val_test_folders(base_dir, train_size=0.6, val_size=0.2, test_size=0.2):
    """
    Splits images into training, validation, and test sets and organizes them into separate subdirectories.
    
    This function takes a base directory containing subdirectories, each corresponding to a class. All images in these
    class directories are then split into training, validation, and test sets according to specified proportions.
    The images are moved into new subdirectories within the base directory, organized by train, validation, and test sets.

    Parameters:
    - base_dir (str): The path to the base directory containing subdirectories of classes with images.
    - train_size (float): The proportion of the dataset to include in the train split (default is 0.6).
    - val_size (float): The proportion of the dataset to include in the validation split (default is 0.2).
    - test_size (float): The proportion of the dataset to include in the test split (default is 0.2).

    Returns:
    None

    Example:
    >>> base_dir = '/path/to/your/dataset'
    >>> create_train_val_test_folders(base_dir)

    Note:
    - The function assumes that the sum of `train_size`, `val_size`, and `test_size` equals 1.0.
    - The directories 'train', 'val', and 'test' will be created under each class directory in the base directory.
    - It is assumed that the base directory is structured such that each subdirectory represents a class and contains images.
    """
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for folder in folders:
        images = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        train, temp = train_test_split(images, train_size=train_size, test_size=(val_size+test_size), random_state=42)
        val, test = train_test_split(temp, train_size=val_size/(val_size + test_size), test_size=test_size/(val_size + test_size), random_state=42)

        # Function to move files
        def move_files(files, dest):
            for f in files:
                shutil.move(f, os.path.join(dest, os.path.basename(f)))

        # Create train, validation, test directories
        os.makedirs(os.path.join(base_dir, 'train', os.path.basename(folder)), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'val', os.path.basename(folder)), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'test', os.path.basename(folder)), exist_ok=True)

        # Move files
        move_files(train, os.path.join(base_dir, 'train', os.path.basename(folder)))
        move_files(val, os.path.join(base_dir, 'val', os.path.basename(folder)))
        move_files(test, os.path.join(base_dir, 'test', os.path.basename(folder)))
