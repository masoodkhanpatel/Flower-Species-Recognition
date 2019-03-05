# Flower Species Recognition

This project consists of a Jupyter notebook to implement an image classifier with PyTorch. Then it has been converted it into a command line application.

The network predicts the species of the flowers.

<img src='assets/Flowers.png' width=80%>

The dataset can be downloaded from [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).

The model has been trained on Google Colab using [`densenet161`](https://pytorch.org/docs/0.3.0/torchvision/models.html) pretrained model with a custom classifier.
```
Flower Species Project.ipynb
```
![](https://raw.githubusercontent.com/resilientmax/Flower-Species-Recognition/master/assets/index.png)

---
## Training

For testing using commandline, we can use the checkpoint we saved in the above notebook.

The first file, `train.py`, will train a new network on a dataset and save the model as a checkpoint. The second file, `predict.py`, uses a trained network to predict the class for an input image. 

Train a new network on a data set with `train.py` in the following ways...

Basic usage: 
```
python train.py data_directory
```
Some options:

- Set directory to save checkpoints: 
```
python train.py data_dir --save_dir save_directory
```
- Choose architecture: 
```
python train.py data_dir --arch "densenet161"
```
- Set hyperparameters: 
```
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
```
- Use GPU for training: 
```
python train.py data_dir --gpu
```
---
## Prediction
Predict flower name from an image with `predict.py` along with the probability of that name. That is, if you pass in a single image `/path/to/image` and return the flower name and class probability.

Basic usage: 
```
python predict.py /path/to/image checkpoint
```
Some options:
- Return top KKK most likely classes: 
```
python predict.py input checkpoint --top_k 3
```
- Use a mapping of categories to real names: 
```
python predict.py input checkpoint --category_names cat_to_name.json
```
- Use GPU for inference: 
```
python predict.py input checkpoint --gpu
```
