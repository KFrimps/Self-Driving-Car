# Simulating a Self-Driving Car

This repository contains the code and data for training a convolutional neural network (CNN) to drive a car in a simulated environment. The project includes data preprocessing, augmentation, model training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

This project implements a self-driving car using a deep learning approach. The car is trained to navigate a track in a simulated environment using a convolutional neural network based on NVIDIA's architecture.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.7+
- TensorFlow 2.0+
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Scikit-learn
- imgaug

You can install the necessary packages using pip:

```sh
pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn imgaug
```
## Dataset

The dataset consists of images captured from three cameras (center, left, and right) mounted on the car, along with corresponding steering angles.

- driving_log.csv: Contains the file paths for the images and their associated steering angles.
- IMG: Directory containing the images captured from the car's cameras.

## Data Preprocessing

The preprocessing steps include:

- Loading the data.
- Extracting file names from the paths.
- Balancing the dataset by reducing the number of samples with a steering angle of zero.
- Splitting the dataset into training and validation sets.

## Data Augmentation

To create a more robust model, several data augmentation techniques are applied:

- Zoom: Randomly zooms into the image.
- Pan: Randomly shifts the image horizontally and vertically.
- Brightness: Randomly changes the brightness of the image.
- Flip: Randomly flips the image horizontally and inverts the steering angle.

## Model Architecture

The model is based on NVIDIA's architecture for self-driving cars, consisting of the following layers:

- Convolutional layers with ELU activation.
- Flattening layer.
- Dense layers with ELU activation.
- Output layer with a single neuron for predicting the steering angle

## Training the Model

The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer. The training process includes:

- Data augmentation using a custom batch generator.
- Training for a specified number of epochs.
- Validation using the validation set.

## Evaluation
The model's performance is evaluated by plotting the training and validation loss over the epochs. The final model is saved as model.h5.

## Usage
- Clone this repository:
```sh
Copy code
git clone https://github.com/KFrimps/Self-Driving-Car.git
cd Self-Driving-Car
``` 
- Ensure you have the necessary data in the specified directory structure.
- Run the Jupyter notebook or Python script to preprocess the data, augment it, train the model, and evaluate the results.

## Results

The training and validation loss are plotted to visualize the model's performance over the epochs. The final model can be used to drive the car in the simulator.


## Acknowledgements

- NVIDIA for the model architecture.
- Udacity for providing the self-driving car simulator. 
