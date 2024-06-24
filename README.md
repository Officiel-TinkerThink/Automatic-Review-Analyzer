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
5. [Recommended papers on Algorithms](#recommended-papers-on-algorithms)
6. [Appendix](#appendix)

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

### Recommended Papers on Algorithms

To understand the foundational and advanced algorithms, you should read the following papers:

1. **Novikoff, A. B. J. (1962). On convergence proofs for perceptrons.**  
   _Symposium on the Mathematical Theory of Automata, 12-16 June, 1962_  
   This paper discusses the convergence proofs for perceptrons, providing foundational insights into their behavior and performance.

2. **Shalev-Shwartz, S., Singer, Y., & Srebro, N. (2007). Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.**  
   _Proceedings of the 24th International Conference on Machine Learning (ICML 2007), pp. 807-814._  
   This paper presents Pegasos, an efficient algorithm for solving Support Vector Machine (SVM) optimization problems using stochastic gradient descent.


### Appendix

#### On Convergence Proofs for Perceptrons - Novikoff (1962)

- **Abstract:** This paper provides the theoretical foundations for the convergence of perceptrons. It demonstrates that perceptrons will converge to a solution in a finite number of steps, given that the data is linearly separable.
- **Key Contributions:**
  - Proof of convergence for the perceptron algorithm.
  - Analysis of the perceptron's behavior under different conditions.
- **Importance:** This work laid the groundwork for understanding the learning dynamics of perceptrons and influenced subsequent research in neural networks and machine learning.

#### Pegasos: Primal Estimated sub-GrAdient SOlver for SVM - Shalev-Shwartz et al. (2007)

- **Abstract:** Pegasos is introduced as a new stochastic gradient descent algorithm for solving the optimization problem posed by Support Vector Machines (SVMs). The paper provides a detailed analysis of its convergence properties and computational efficiency.
- **Key Contributions:**
  - Development of the Pegasos algorithm.
  - Empirical evaluation demonstrating the efficiency and scalability of Pegasos.
  - Theoretical analysis proving the convergence rate of the algorithm.
- **Importance:** Pegasos significantly improves the efficiency of SVM training, making it feasible to apply SVMs to large-scale machine learning problems.






