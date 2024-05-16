from utils.download_data_fromkaggle import *
import warnings
warnings.filterwarnings("ignore")
from main_edit import * 

"""
FOLDERS: 
        - Soil types
        - CUB
"""

folder = "Soil types"
img_root = "/home/lorenzo/Desktop/introml/intromlproject/C:/Users/Mardeen/Desktop/fine-grained-imgs/data/C:/Users/Mardeen/Desktop/fine-grained-imgs/data/Soil types"

"""
MODELS: 
        alexnet, efficientnet, inceptionv4, inceptionv4_freeze, inceptionv3, inceptionv3_freeze
"""
model_name = "efficientnet"


###################################
print("done----")
transform = transform_dataset(resize=(256,256),
                  crop=(224,224),
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])

train_dataset, val_dataset, test_dataset = read_dataset(img_root, transform=transform)
num_classes = len(train_dataset.classes)

main(run_name= f"{model_name}_{folder}",
         batch_size=128,
         device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
         checkpoint_path = f'./models/checkpoint_{model_name}.pth',
         learning_rate=0.001,
         weight_decay=0.000001,
         momentum=0.9,
         epochs=10,
         criterion = nn.CrossEntropyLoss(),
         visualization_name= f"{model_name}",
         img_root= img_root,
         save_every=1,
         init_model = init_model(model_name, num_classes),
         transform = transform_dataset())


#%tensorboard --logdir logs/fit