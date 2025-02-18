# Credit Card Fraud Detection

## Project Overview

This project aims to build a machine learning model to detect fraudulent credit card transactions. The dataset used contains anonymized transaction details, and the goal is to classify transactions as fraudulent or non-fraudulent using various machine learning algorithms.

## Dataset

The dataset used is the creditcard.csv file, which contains:

Time: Seconds elapsed between each transaction and the first transaction in the dataset.

V1 to V28: Anonymized features resulting from a PCA transformation.

Amount: Transaction amount.

Class: Target variable (0 for non-fraudulent, 1 for fraudulent transactions).

Dataset Link

Download the dataset here

Dependencies

To run this project, install the following Python libraries:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost

Exploratory Data Analysis (EDA)

Checked for missing values and duplicates.

Performed statistical analysis on numerical features.

Created visualizations: histograms, boxplots, violin plots, density plots, and pair plots.

Analyzed categorical features.

Conducted multivariate analysis using scatter plots and correlation heatmaps.

Data Preprocessing

Outliers detected and handled using the IQR method.

Data scaled using StandardScaler.

Dataset split into training and testing sets (80%-20%).

Machine Learning Models

The following models were trained and evaluated:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

AdaBoost Classifier

XGBoost Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Naive Bayes (BernoulliNB)

Hyperparameter Tuning

Used GridSearchCV and RandomizedSearchCV to optimize the following models:

Decision Tree Classifier

Random Forest Classifier

AdaBoost Classifier

Gradient Boosting Classifier

XGBoost Classifier

Support Vector Machine

K-Nearest Neighbors

Model Evaluation

Each model was evaluated using:

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

ROC-AUC Score & Curve

Results Summary

The models' performance was compared based on accuracy and ROC-AUC score. The best-performing model was selected for deployment.

Future Improvements

Implement deep learning models such as Neural Networks.

Use advanced anomaly detection techniques.

Experiment with feature engineering to improve classification performance.

Usage

Run the following command to execute the fraud detection script:

python fraud_detection.py

This will train models and display evaluation metrics.

Project Deployment

View the deployed project here

Author

Pranav Atkane

