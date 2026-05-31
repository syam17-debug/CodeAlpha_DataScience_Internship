# Car Price Prediction using Machine Learning

## Overview

This project predicts the selling price of a car using Machine Learning techniques. A Random Forest Regressor model is trained on historical car data containing features such as year of manufacture, present price, kilometers driven, fuel type, transmission type, and ownership details.

The goal is to build a regression model capable of accurately estimating a car's selling price based on its characteristics.

## Dataset

The dataset contains information about used cars, including:

* Car Name
* Year
* Selling Price (Target Variable)
* Present Price
* Kilometers Driven
* Fuel Type
* Selling Type
* Transmission
* Owner

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

## Project Workflow

### 1. Data Collection

The car dataset is loaded using Pandas.

### 2. Data Preprocessing

* Removed unnecessary columns such as Car_Name.
* Converted categorical variables into numerical format using one-hot encoding.
* Created machine-learning-ready features.

### 3. Feature Selection

Input features include:

* Present Price
* Kilometers Driven
* Fuel Type
* Selling Type
* Transmission
* Owner
* Year/Car Age

Target Variable:

* Selling_Price

### 4. Model Training

A Random Forest Regressor is trained using the training dataset.

### 5. Model Evaluation

The model is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

## Results

Model Performance:

* MAE: 0.61
* RMSE: 0.93
* R² Score: 0.9627

The model achieved approximately 96% prediction accuracy, indicating excellent performance on unseen data.

## Visualizations Generated

### Price Distribution

Shows the distribution of selling prices in the dataset.

### Actual vs Predicted Prices

Compares actual car prices with model predictions.

### Feature Importance

Displays the most influential features affecting car prices.

## Project Structure

Task_3_Car_Price_Prediction/

├── car_data.csv

├── carprice.py

├── requirements.txt

├── README.md

├── report.pdf

└── images/

    ├── price_distribution.png

    ├── actual_vs_predicted.png

    └── feature_importance.png

## Conclusion

This project demonstrates the practical application of Machine Learning in predicting vehicle prices. The Random Forest Regressor successfully captures the relationship between vehicle features and selling price, achieving high prediction accuracy and providing valuable insights into the factors affecting car valuation.
