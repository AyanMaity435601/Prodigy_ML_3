# Cat-Dog Classification

## Overview
Here I used Support Vector Classifier to classify cats and dogs.

## Script Explanation
The script 'cat_dog_classification.py' performs the following steps:

Load images from the specified folders and resize them to 64x64 pixels.
Flatten the images and create labels (0 for cats, 1 for dogs).
Split the dataset into training and validation sets.
Apply PCA to reduce the dimensionality of the data.
Train an SVM classifier on the PCA-transformed training data.
Evaluate the classifier on the validation set.
Load the test images and evaluate the classifier on the test set.

## Dataset
The dataset used for this project is the [Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats?select=train) from Kaggle. 
