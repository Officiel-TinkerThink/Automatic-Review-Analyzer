# Automatic-Review-Analyzer
This Project of Machine Learning is created in order to fulfill project requirements provided by MITx

# Perceptron-Based Sentiment Analysis Project

## Overview

This project focuses on implementing and comparing three variants of the Perceptron algorithm to classify the sentiment of textual reviews. The task is to determine whether a given review is positive or negative based on the text content. The three variants implemented are:

- **Perceptron**
- **Average Perceptron**
- **Pegasos (Primal Estimated sub-GrAdient SOlver for SVM)**

## Table of Contents

1. [Introduction](#introduction)
2. [Algorithms](#algorithms)
3. [Usage](#usage)
4. [Future Work](#future-work)
5. [Contributing](#contributing)

## Introduction

Sentiment analysis is a common application of natural language processing (NLP) where the goal is to determine the sentiment expressed in a piece of text. This project utilizes the Perceptron algorithm and its variants to train models that can classify review sentiments as either positive or negative.


## Algorithms

### 1. Perceptron

The basic Perceptron algorithm is a type of linear classifier. It updates its weights based on the errors made during the prediction. It is simple yet effective for linearly separable data.

### 2. Average Perceptron

The Average Perceptron is an extension of the basic Perceptron. It maintains an average of the weights over all updates, which tends to yield better generalization on unseen data.

### 3. Pegasos

Pegasos stands for Primal Estimated sub-GrAdient SOlver for SVM. It is an efficient algorithm for training support vector machines (SVM) using stochastic gradient descent (SGD). Pegasos is known for its simplicity and strong theoretical guarantees.


## Usage

To train and evaluate the models, follow these steps:

1. **Preprocess the Data:**  
   Run the data preprocessing notebook to read and prepare the dataset for training.

2. **Train the Models:**  
   Execute the model training notebook to train the Perceptron, Average Perceptron, and Pegasos models on the training data.

3. **Evaluate the Models:**  
   The trained models are evaluated on the test set to determine their accuracy and other performance metrics. The results are saved in the `results/` directory.

## Results

The result of the experiment is the accuracy for each algorithm. These results provide insights into the performance of each variant on the sentiment analysis task.

## Future Work

Possible extensions to this project include:

- Implementing additional NLP preprocessing steps to improve model performance.
- Experimenting with other machine learning algorithms such as logistic regression, SVM, or neural networks.
- Exploring more complex sentiment analysis tasks with multi-class classification or aspect-based sentiment analysis.

## Contributing

Contributions are welcome! If you have any improvements or new ideas, please fork the repository and submit a pull request. Make sure to follow the project's coding style and include appropriate tests for your changes




