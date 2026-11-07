# Spaceship Titanic Prediction
![image](spaceship_banner.png)

A CS 300 Introduction to Machine Learning midterm project by Htet Khant Linn. The project uses Kaggle’s Spaceship Titanic dataset to predict whether a passenger was transported to another dimension after the ship encountered a spacetime anomaly.

## Dataset and model

The [Spaceship Titanic dataset](https://www.kaggle.com/competitions/spaceship-titanic/data) contains 8,693 labeled training records. The target column is `Transported`.

I compared Logistic Regression and K-Nearest Neighbors (KNN) using stratified 5-fold cross-validation. Both pipelines included missing-value imputation, scaling, one-hot encoding, and `SelectKBest` feature selection. The tuned KNN model achieved the highest mean cross-validation accuracy at **79.38%** and reached **78.61% accuracy**.

## Streamlit app

[Click Here to Test App](https://spaceship-titanic-hkl.streamlit.app/)
Enter a passenger’s details to predict their transport status, or browse the original training dataset in an interactive table.
