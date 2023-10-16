# CIFAR10-CNN-Build

# Project Title: Optimising Convolutional Neural Networks (CNNs) for CIFAR-10 Classification

## Description:

This Python script is designed for building and evaluating Convolutional Neural Network (CNN) models on the CIFAR-10 dataset using the TensorFlow and Keras libraries. The project involves a series of experiments aimed at optimising the model's architecture and hyperparameters. Below, I'll provide an overview of the code and discuss some of the key results and findings.

### Code Overview:

#### Importing Libraries: The script begins by importing necessary libraries, including TensorFlow, NumPy, math, timeit, and Matplotlib. These libraries are essential for data manipulation, visualisation, and machine learning.

#### Loading Dataset: The code loads the CIFAR-10 dataset, comprising 60,000 32x32 color images categorized into 10 classes. The dataset is divided into 50,000 training images and 10,000 test images. Data preprocessing involves normalising the pixel values to enhance model training.

#### Model Architecture: The primary focus of the code is to experiment with various CNN architectures and hyperparameters to maximize classification accuracy. It presents several experiments to investigate the impact of different architectural variations, optimisers, activation functions, dropout rates, and regularization techniques. The notable experiments are as follows:

* Base Model: The initial model serves as a baseline for further experimentation.

* Experiment 1: Increases the number of convolutional blocks in the model.

* Experiment 2: Replaces max-pooling with average-pooling to assess the effect on accuracy.

* Experiment 3: Adjusts the activation functions in the hidden layers to tanh or LeakyReLU.

* Experiment 4: Replaces the RMSprop optimizer with Adam or SGD using different learning rates to explore their impact.

* Experiment 5: Adds dropout layers to mitigate overfitting. Various dropout rates are experimented with.

* Experiment 6: Introduces L2 regularisation in combination with different dropout values.

* Experiment 7: Incorporates batch normalisation along with increased dropout values.

* Experiment 8 - Mini Batch: In this experiment, the code examines the impact of batch size by changing it to 16. This batch size demonstrates a near-perfect balance between train and test accuracy, with minimal overfitting.

* Experiment 8.1 - Increase Mini Batch Size: This sub-experiment increases the batch size to 64.

* Experiment 9 - Orthogonal Regularisation: A new type of kernel regularisation, Orthogonal Regularisation, is introduced. The code investigates its effectiveness compared to other regularisers.

* Experiment 10 - More Epochs with Early Stopping: This experiment aims to find the highest achievable accuracy by training for more epochs while incorporating early stopping to prevent overtraining.

For each experiment, the code defines a specific CNN model, compiles it, trains it on the training data, and evaluates its accuracy on the test dataset. The results of each experiment, including test accuracy, are printed to the console.

### Key Findings and Results:

The code provides a structured approach to systematically explore different architectural variations, optimisers, activation functions, dropout rates, and regularisation techniques. The goal is to identify the combination that yields the highest accuracy on the CIFAR-10 dataset.

* Visualisations: The code provides various visualisations, including layer-by-layer architecture diagrams and activation maps, making it easier to understand the model's inner workings.

* Accuracy Plots: The script includes plots of loss, confusion matrix, and accuracy to visually analyze model performance across different experiments.

* Correctly and Incorrectly Predicted Images: The code presents visually appealing images, both correctly and incorrectly classified by the model.
