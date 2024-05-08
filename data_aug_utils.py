import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img

def rotate_image(img, degrees=90):
    """Rotate the given image by the specified degrees.
    Must be a multiple of 90.
    """
    if degrees % 90 != 0:
        raise ValueError("degrees must be a multiple of 90.")
    k = degrees // 90
    return tf.image.rot90(img, k=k)

def flip_image(img, mode='horizontal'):
    """Flip the image either horizontally or vertically."""
    if mode == 'horizontal':
        return tf.image.flip_left_right(img)
    elif mode == 'vertical':
        return tf.image.flip_up_down(img)
    else:
        raise ValueError("mode must be 'horizontal' or 'vertical'.")

def zoom_image(img, zoom_range=(0.8, 1.2)):
    """Zoom the image within the specified range. 
    Assumes the image is square.
    """
    if img.shape[0] != img.shape[1]:
        raise ValueError("Zoom function assumes a square image.")
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    height, width = img.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    img = tf.image.resize(img, (new_height, new_width))
    return tf.image.resize_with_crop_or_pad(img, height, width)

def color_adjust(img, brightness=0, contrast=1):
    """
    Adjust the brightness and contrast of the image.
    
    Parameters:
    - brightness (float): A number between -1 and 1, where 0 means no change.
      Positive values increase the brightness, negative values decrease it.
    - contrast (float): A number, where 1 means no change. Values greater than 1
      enhance contrast, and less than 1 reduce contrast. Practical upper limit is around 3.
    
    Returns:
    - A TensorFlow Tensor representing the adjusted image."""
    img = tf.image.adjust_brightness(img, delta=brightness)
    return tf.image.adjust_contrast(img, contrast_factor=contrast)

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

def get_data_file(folder_name, dataset_url):
    """Download and extract a dataset, returning the local directory path."""
    if not dataset_url:
        raise ValueError("Dataset URL cannot be empty.")
    return tf.keras.utils.get_file(
        folder_name, 
        origin=dataset_url, 
        cache_dir='.', 
        untar=True
    )

def save_augmented_images(images, labels, output_dir, class_indices):
    """Save augmented images to labeled directories."""
    # Reverse class indices to get label names from indices
    label_names = {v: k for k, v in class_indices.items()}
    for idx, (image, label) in enumerate(zip(images, labels)):
        # Get class label
        label_name = label_names[label.argmax()]
        class_dir = os.path.join(output_dir, label_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        # Save image
        img = array_to_img(image)
        img.save(os.path.join(class_dir, f'aug_{idx}.png'))