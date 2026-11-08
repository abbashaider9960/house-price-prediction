# California Housing Price Prediction

## Project Overview

This project uses Machine Learning to predict median house values in California using a Random Forest Regressor.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Dataset

California Housing dataset from Scikit-learn, containing 20,640 records and 8 input features.

## Model Performance

| Metric | Result |
|---|---|
| MAE | $32,773.11 |
| RMSE | $50,598.60 |
| R² Score | 0.8046 |

## Features

- Data exploration and preprocessing
- Train/test split
- Random Forest model training
- Model evaluation
- Saving and loading the trained model
- Predicting median house values using user inputs

## How to Run

1. Install dependencies:

   `python -m pip install -r requirements.txt`

2. Train and save the model:

   `python main.py`

3. Load the saved model and make predictions:

   `python load_model.py`

## Note

This model predicts median house values for California census areas, not exact prices of individual properties.