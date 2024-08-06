# FinanceMood

FinanceMood is a sentiment analysis project aimed at analyzing financial news headlines to predict their impact. The project uses various models and techniques to classify news sentiments into categories such as negative, neutral, or positive.

## Project Overview

The `FinanceMood` repository contains code to train and evaluate different machine learning models for sentiment analysis on financial news headlines. The project includes:

1. **`model.py`**: A baseline model using a standard neural network architecture.
2. **`model_attention.py`**: An advanced model utilizing Multi-Head Attention for better context understanding.
3. **`model_tuning.py`**: A hyperparameter tuning script using Keras Tuner to optimize model performance.


## Dataset
https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news \
The dataset used in this project is the Sentiment Analysis for Financial News dataset from Kaggle. It contains financial news headlines and their corresponding sentiments.

**Content:**

1. **Sentiment**: The sentiment label, which can be negative, neutral, or positive.
2. **News Headline**: The text of the financial news headline.
