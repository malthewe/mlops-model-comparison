# Titanic MLOps Project 

This project demonstrates the use of **MLflow** for experiment tracking and comparison of different classification models on the Titanic dataset.

## Project Overview

I use the Titanic dataset from Seaborn and compare three machine learning models:

- Random Forest
- Logistic Regression
- Gradient Boosting

All models are logged using **MLflow**, including hyperparameters and performance metrics such as accuracy, precision, and recall.

## Dataset

The dataset contains information about Titanic passengers, including:

- Survival (`survived`)
- Class (`pclass`)
- Gender (`sex`)
- Age (`age`)
- Number of siblings/spouses (`sibsp`)
- Number of parents/children (`parch`)
- Ticket fare (`fare`)
- Port of embarkation (`embarked`)

## Metrics

Each model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall

## MLflow

MLflow is used to:

- Track experiments
- Log model parameters and metrics
- Save trained models
