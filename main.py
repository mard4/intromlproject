from utils.download_data import *
from utils.epochs_loop import * 

# !!! IMPORTANT: DO NOT DELETE ANY OF THE VARIABLES BELOW, JUST EDIT THE VALUES !!!

"""
FOLDERS: 
        - Soil types
        - CUB
"""

# !!! edit: Soil types è la cartella che contiene le cartelle train, val, test perché abbiamo scaricato questo dataset
# , altrimenti sarebbe stata CUB o altro
folder = "Soil types"
# !!! edit: img_root è il path della cartella che contiene le cartella train, val, test, settatelo con il path corretto
path = "/home/disi/cartella1/intromlproject/"

img_root = path + folder

"""
MODELS: 
        alexnet, efficientnet, inceptionv4, inceptionv4_freeze, inceptionv3, inceptionv3_freeze, densenet201
"""
model_name = "densenet201"

transform = transform_dataset(resize=(256,256),
                  crop=(224,224),
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])

learning_rate = 0.001
weight_decay = 0.000001
momentum = 0.9
betas=(0.9, 0.999)
epochs = 10
criterion  = nn.CrossEntropyLoss()
batch_size = 60        #128
# Optimizer type can be either "Custom" or "Fixed" see optimizer.py for more details
optimizer_type = "Custom"  
# Optimz can be:     optim.Adam, optim.SGD, optim.AdamW 
optimz = torch.optim.Adam        
param_groups = [{'prefixes': ['classifier'], 'lr': learning_rate * 10},
                {'prefixes': ['features']}]


#########################################  DO NOT EDIT BELOW THIS LINE  #########################################
train_data, _, _ = read_dataset(img_root, transform=transform)
num_classes = len(train_data.classes)
main(run_name= f"{model_name}_{folder}_{optimizer_type}_{str(optimz)}",
         batch_size = batch_size,
         device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
         checkpoint_path = f'{path}models/checkpoint_{model_name}.pth',
         learning_rate=learning_rate,
         weight_decay=weight_decay,
         momentum=momentum,
         betas=betas,
         epochs=epochs,
         criterion = criterion,
         optimizer_type = optimizer_type,        # "Custom" or "Fixed"
         optim = optimz,                        #optim.Adam, optim.SGD, optim.AdamW, etc.
         param_groups = param_groups,      # set to None if not using custom optimizer
         visualization_name= f"{model_name}",
         img_root= img_root,
         save_every=1,
         init_model = init_model(model_name, num_classes),
         transform = transform_dataset())


#%tensorboard --logdir logs/fit