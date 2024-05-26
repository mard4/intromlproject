# Project on Fine-Grained Image Classification

<div id="header" align="center">
  <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="80"/>
</div>
This project is part of a Machine Learning course at the University of Trento.
Fine-Grained Image Classification is a task in computer vision where the goal is to classify images into subcategories within a larger category. For example, classifying different species of birds or different types of flowers. This task is considered to be fine-grained because it requires the model to distinguish between subtle differences in visual appearance and patterns, making it more challenging than regular image classification tasks. 

<div style="text-align: center;"align="center">
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>
  <img src="https://github.com/devicons/devicon/blob/master/icons/git/git-original-wordmark.svg" title="Git" alt="Git" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/pytorch/pytorch-original.svg" title="PyTorch" alt="PyTorch" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/tensorflow/tensorflow-original.svg" title="TensorFlow" alt="TensorFlow" width="40" height="40"/>
</div>

### Table of Contents

- <a href='#installation'>Installation</a>
- <a href='#structure'>Structure</a>
- <a href='#usage'>Usage</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#models'>Models</a>
- <a href='#reference'>Reference</a>
- <a href='#authors'>Authors</a>

## Installation
<div style="text-align: center;">
  <p>This project has been tested on an Azure machine running Ubuntu.</p>
  <div style="float: right;">
    <img src="images-readme/neofetch.png" alt="neofetch" width="400"/>
  </div>
</div>

Create a folder (I suggest you to call it <b>"ml"</b> to avoid editing more paths), navigate inside it and clone the repository. Create a virtual environment, activate it  `source .venv/bin/activate `, then run `pip install -r requirements.txt `.

Note that you should also have an nvidia gpu and cuda drivers installed, otherwhise the models will be loaded in cpu and will be much slower and prone to crashes.

## Structure

The starting configuration will look something like this:

- virtualenvironment

- intromlproject
  
  - requirements.txt
  
  - data_setupper.ipynb $\to$ notebook used to download and prepare data
  
  - training-main.py $\to$ python script that trains and validates the chosen model
  
  - testing-main.py $\to$ python script that tests the provided model.
  
  - testing-main-exam.py $\to$ python script that tests a model and 
  
  - utils/
    
    - augmentation.py $\to$ contains functions for data augmentation.
    
    - custom_models.py $\to$ contains functions to built models from scratch.
    
    - downloader.py $\to$ contains functions to download data from different sources.
    
    - extractors.py $\to$ contains functions to extract different archived files
    
    - logger.py $\to$ contains a function which initializes the logger.
    
    - models_init.py $\to$ contains functions to initialize models from already existing libraries.
    
    - optimizers.py $\to$ contains a function to implement a custom optimizer. #todo
    
    - testing.py $\to$ contains the function to test a model and send files to the competition server.
    
    - training.py $\to$ contains the functions to train and validate the model for a single epoch. 
    
    - testing.py $\to$ contains the function to test a model storing the predictions in a dictionary and submits it to a server.

## Usage

#### Datasets

Navigate inside the repository, then you can use the `data_setupper.ipynb` notebook to download any dataset of your choosing. We decided to use a notebook instead of a .py file because the datasets we tried came from different sources (kaggle, torch, generic websites) and we wanted to have the flexibility to just download them and store them inside a datasets folder, which will be automatically created during your first download. 

All you have to do is follow the comments `@edit` and edit accordingly to your computers paths. If you called the parent folder "ml", then some paths are already correct. More information is available inside the notebook. If you followed the procedure correctly you will have the following datasets configuration:

<div style="text-align: center;">
  <div style="float: right;">
    <img src="images-readme/folders.jpg" alt="neofetch" width="350"/>
  </div>
</div>

Only the name of the specific dataset and the name of the folders containing the images will differ based on what you downloaded.

#### Model training and testing

#### Training

You can check the `model_initializers` dictionary inside the `init_model` function in the `models_init.py` to see all the models implemented, copy and paste the name of the model you want to use, then open the `training-main.py` to set the training up.

Paste in the variable `model_name` what you just copied, then edit the variables `root`, `img_folder` accordingly to your specific system path. Referring to the graph above, `img_folder` should be named as `CUB200-2011` or `Flowers102` etc, lastly `root` should be the system path up until the folder I suggested to you to call `ml`, which is the folder that contains the virtual environment, the datasets and the repository.

After this, edit the files inside the config however you desire, the f-strings will allow for a smoother experience automatically constructing the correct paths for everything that will be tracked during the training.

Once you have set the training up, you can just run the file from the terminal (remember to have the virtual environment activated).



After you trained your model, there will be a new folder which will contain the weights saved as a .pth file. If you wish to keep training the model, you can just edit the `training-main.py` file adding the checkpoint instead of leaving it to None. You can keep repeating this until you are satisfied with your results.


#### Testing
Now that our model is ready, we can test it. 

Open the `testing-main.py` and notice how the file is extremely similar to the training one. The most important thing is to make sure to select the correct model and path to your save file (a file ending in .pth) to load its weights. Make sure that the test dataset is loaded correctly. Then just run the file from the virtual environment like we did for the training one.  

## Reference

## Authors
- [Lorenzo Chicco](https://github.com/lorenzochicco99/)
- [Martina D'Angelo](https://github.com/mard4/)
- [Enrico Guerriero](https://github.com/enricoguerriero/)
