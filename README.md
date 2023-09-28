# Subscription Churn Prediction Challenge

## Overview

Subscription services are an integral part of various industries, and retaining subscribers is a crucial business goal. This challenge focuses on predicting user churn in subscription-based businesses using machine learning techniques. Churn prediction helps companies identify users at high risk of canceling their subscriptions, allowing for targeted retention strategies.

In this project, we will walk through the following steps:
1. Exploratory Data Analysis (EDA)
2. Random Forest Classifier
3. Cross-Validation
4. Bagging Classifier

## Getting Started

### Prerequisites

Before running the code, ensure you have the following packages installed:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Installation

You can install these packages using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Data

The dataset used for this challenge should be described in your code or documentation. Ensure that you load and preprocess the data appropriately.

## Exploratory Data Analysis (EDA)

EDA is essential to understand the dataset's characteristics and relationships between variables. In your code, you should have performed the following EDA tasks:
- Data summary statistics
- Data visualization using matplotlib and seaborn
- Handling missing values
- Encoding categorical variables

## Random Forest Classifier

The Random Forest algorithm is a powerful tool for classification tasks. In this section, you should:
- Split your data into training and testing sets
- Train a Random Forest Classifier on the training data
- Evaluate the model using accuracy, classification report, confusion matrix, and ROC AUC score
- Interpret the results and discuss model performance

## Cross-Validation

Cross-validation is a crucial step to assess the model's generalization performance. You should:
- Use cross-validation techniques (e.g., k-fold cross-validation) to evaluate the Random Forest model
- Calculate the mean accuracy and ROC AUC score across folds
- Discuss whether the model is overfitting or underfitting

## Bagging Classifier

Bagging is an ensemble technique that can improve model stability. In this section, you should:
- Implement a Bagging Classifier, which uses a base classifier (e.g., Decision Tree)
- Train and evaluate the Bagging Classifier using the same metrics as the Random Forest model
- Compare the performance of the Bagging Classifier with the Random Forest Classifier
- Discuss any advantages or disadvantages of using bagging in this context

## Conclusion

The Bagging Classifier demonstrated promising results in predicting subscription churn, with an accuracy of 0.82 and an AUC score of 0.73. These metrics suggest that the model is performing well in identifying users at risk of churning.
