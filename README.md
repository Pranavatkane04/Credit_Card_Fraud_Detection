# Credit Card Fraud Detection

## Project Overview

This project aims to build a machine learning model to detect fraudulent credit card transactions. The dataset used contains anonymized transaction details, and the goal is to classify transactions as fraudulent or non-fraudulent using various machine learning algorithms.

## Dataset

The dataset used is the creditcard.csv file, which contains:

Time: Seconds elapsed between each transaction and the first transaction in the dataset.

V1 to V28: Anonymized features resulting from a PCA transformation.

Amount: Transaction amount.

Class: Target variable (0 for non-fraudulent, 1 for fraudulent transactions).

### Dataset Link

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dependencies

To run this project, install the following Python libraries:

pip install numpy ,pandas ,matplotlib ,seaborn ,scikit-learn ,xgboost

## Exploratory Data Analysis (EDA)

Checked for missing values and duplicates.

Performed statistical analysis on numerical features.

Created visualizations: histograms, boxplots, violin plots, density plots, and pair plots.

Analyzed categorical features.

Conducted multivariate analysis using scatter plots and correlation heatmaps.

## Data Preprocessing

Outliers detected and handled using the IQR method.

Data scaled using StandardScaler.

Dataset split into training and testing sets (80%-20%).

Machine Learning Models

### The following models were trained and evaluated:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

AdaBoost Classifier

XGBoost Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Naive Bayes (BernoulliNB)

### Hyperparameter Tuning

Used GridSearchCV and RandomizedSearchCV to optimize the following models:

Decision Tree Classifier

Random Forest Classifier

AdaBoost Classifier

Gradient Boosting Classifier

XGBoost Classifier

Support Vector Machine

K-Nearest Neighbors

## Model Evaluation

Each model was evaluated using:

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

ROC-AUC Score & Curve

## Results Summary

The models' performance was compared based on accuracy and ROC-AUC score. The best-performing model was selected for deployment.

## Future Improvements

Implement deep learning models such as Neural Networks.

Use advanced anomaly detection techniques.

Experiment with feature engineering to improve classification performance.

## Usage

Run the following command to execute the fraud detection script:

python fraud_detection.py

This will train models and display evaluation metrics.

## Project Deployment

https://creditcardfrauddetectiongit-kh76ejarr8mlppmgtwdgex.streamlit.app/

## Screenshots

![{E59D3844-0FB6-4E83-923A-0AC416BBE6C8}](https://github.com/user-attachments/assets/fefe6a1b-03cd-4e81-9045-4ec4b56fb4d6)

![{05A05CEC-CD65-4447-812A-E69121EEFBC9}](https://github.com/user-attachments/assets/f671873e-9dde-481e-9043-ec1c4c43edf4)

![{568569E2-733F-4D28-B8D1-5C0E467A7B72}](https://github.com/user-attachments/assets/c174a4ae-2441-4811-9d75-03f9275a0a87)

![{AB87C8BA-9D9A-4E4A-BE03-B9CF40010B20}](https://github.com/user-attachments/assets/7f9a64d7-8a43-4e8b-847e-b00e87a1ab4a)

![{D243E5BC-A580-4A8C-87D7-D4FCEBF6A7F5}](https://github.com/user-attachments/assets/7e01167d-00b0-47a9-aaf1-301433776e33)






## Author

Pranav Atkane

