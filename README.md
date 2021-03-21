# Neural Network for Image Feature Prediction

Codebase for mapping two-photon calcium imaging neuron response data to frames in a 
movie stimulus in order to decode the link between sensory encoding and perception. 

Code can be used by following the 2 jupyter notebooks in order to:

* Notebook 1: preprocess_and_Gaborwavelets.ipynb
    * smooth and normalize calcium response data
    * downsample and normalize the movie frames
    * generate a bank of Gabor filter wavelets for image decomposition
    * decompose preprocessed movie frames into feature vectors and vice-versa
    
* Notebook 2: train_test_nn.ipynb
    * intialize and train neural network model to learn the relationship between 
        responses and image features
    * test the model performance using mean-squared-errors and correlations between 
        predicted and ground-truth images
    * view predicted frames from arbitrary or artificially perturbed response vectors

To use, first either create a new environment and pip install the packages listed in 
requirements.txt or install these packages into your working environment using

`pip install -r requirements.txt`

It is HIGHLY RECOMMENDED to create a new environment before installing the required
packages and running the notebooks. 

To run the notebooks, open jupyter and follow the steps for defining save/load paths
and defining necessary parameters at various steps. 

Input data for the preprocessing notebook requires pre-generated data sets consisting of

1) response traces for every cell in a numpy array (.npz file) of shape (nCells,nMovieReps,nFramesPerRep)
2) .mp4 file of movie

author: Paul LaFosse

contact: lafossek21@gmail.com
