from utils.download_data_fromkaggle import *
from utils.data_aug_utils import *
import warnings
warnings.filterwarnings("ignore")

"""
    define url, folder name and download dataset
"""

data_path = "."
folder_name = "C:/Users/Mardeen/Desktop/fine-grained-imgs/data"   # scarica dove stai lavorando ora
folder = "/Soil types"
dataset_url = "https://www.kaggle.com/datasets/prasanshasatpathy/soil-types"
img_root = "C:/Users/Mardeen/Desktop/fine-grained-imgs/data/Soil types"

# !!! if you're downloading a dataset from Kaggle
download_dataset_kaggle(dataset_url, folder_name)

# !!! if you're downloading a file from any URL
#img_root = get_data_file(folder_name, dataset_url)

# !!!! if you're downloading a ZIP file, use also this
#img_root = extract_tgz(download_dataset_kaggle(dataset_url, folder_name), extract_to=None)
print("img root", img_root)

# !!! Crea le cartelle train, val, test
create_train_val_test_folders(img_root, train_size=0.6, val_size=0.2, test_size=0.2)   

# !!! Data Augmentation
image_generator =  setup_data_generator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

directory_path = folder_name + folder + "/train"
train_generator = load_data_from_directory(directory_path, target_size=(150, 150), batch_size=32)

n_of_batches = 50

# Base directory where each class has its own subdirectory
for i, (images, labels) in enumerate(train_generator):
    if i >= n_of_batches:  # Stop after saving images from 50 batches
        break
    save_augmented_images(images, labels, directory_path, train_generator.class_indices)


