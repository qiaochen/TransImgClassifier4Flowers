### Data

The data for this project is from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html. There are 102 different types of flowers, where there are ~20 images per flower to train on.

The dataset should be split into train, valid, and test sets respectively, where the flower images are organized within the same folder if they belong to the same class, and the folder is named with the class id.


### Specifications
The file `train.py` will train a new network on the dataset and save the model as a checkpoint. The file `predict.py`, uses a trained network to predict the class for an input image. 

#### 1. Train a new network on a data set with `train.py`

- Basic usage: `python train.py data_directory` Prints out training loss, validation loss, and validation accuracy as the network trains
- Options:
    - Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
    - Choose architecture: `python train.py data_dir --arch "vgg13"`
    - Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
    - Use GPU for training: `python train.py data_dir --gpu`

#### 2. Predict flower name from an image with predict.py
Predicion would also return the probability of that name. To predict an image picture, you'll pass in a single image /path/to/image and return the flower name and class probability.

- Basic usage: `python predict.py /path/to/image trained_model.pth`
- Options:
    - Return top KK most likely classes: `python predict.py image_path trained_model.pth --top_k 3`
    - Use a mapping of categories to real names: `python predict.py image_path trained_model.pth --category_names cat_to_name.json`
    - Use GPU for inference: `python predict.py image_path trained_model.pth --gpu`
    
> You may also consider viewing `Image Classifier Project.ipython` to get an intuitive understanding on what has been done in this project.

