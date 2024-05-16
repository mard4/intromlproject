# readme
### Table of Contents

- <a href='#requirements'>Requirements</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training L-Net'>Training</a>
- <a href='#evaluation'>Evaluation</a>
- <a href='#model'>Model</a>
- <a href='#reference'>Reference</a>

## Requirements
Run pip install -r requirements.txt in bash or powershell in your virtual environment. Our project was originally on Ubuntu 24.04 etc

## How is the data organized?

main folder -> train,test,val -> classes -> images
## Instructions
- Prepare_folders.py is a file where you can find download functions, either for kaggle (remember to have your kaggle.json file located in folder etc) or from a direct download link. We are also creating the train-validation-test folders, and creaitng augmented images. Prima trasformi e poi generi o prima generi e poi trasformi?
- main.py is the main file where you will need to define the main folder of the dataset you will use this repoository on. You can find the list inside the main.py file (add the exact lines). There is also a list of models we tried, pick the one you desire and set it as model_name. Define the trasformation and its parameters, note that each model has a preferred image size, you can find the optimal image size per model in the init_models.py file. After that, you can choose all hyperparameters and other parameters (ca capire come chiamarli). After this, you can execute the main.py which will initialize the training loop.
- ? come facciamo a caricare dei pesi?
- come facciamo a runnare il test loop?
- da dove faccio partire i check point?
- aggiustare validation e train loop per implementare i checkpoints e scegliere se usare o meno il validation loop durante la fase di traianing 
- aggiustare la funzione criterion
- modificare i checkpoint per salvare la migliore epoca (paragonare train-val loss alla precedente)
- docstring per le varie funzioni
- per i transformer, c'è il dsicrimantor, serve un loop diverso?



## Reference