# Advanced Machine Learning Project 1: Brain Age Prediction

## Introduction
This project was conducted for the course "Advanced Machine Learning" that I followed in Fall 2022 at ETHZ: https://ml2.inf.ethz.ch/courses/aml2022/


## Goal
The goal was to predict a person's age from brain image data measured by an MRI scan. The original dataset included 832 features as well as multiple outliers and NaN values.

## Solution
1. Outlier removal
   - Remove outliers with the 3-sigma rule
   - Replace NaN missing values with k-Nearest Neighbors imputer
   - Isolation Forest Algorithm with a contamination of 20%, and a max sample of 100

2. Preprocessing and feature selection
  - Reduce the number of features to 200 using the SelectKBest function from scikit with a univariate linear regression for the classification method.
  - Lasso Regression using GridSearchCV (to find the best hyperparameter) to further reduce the number of features to 185
    
3. Model selection
   - Used StackingRegressor from sci-kit with the following estimators : Lasso, XGBoost, AdaBoost, Random Forest.
