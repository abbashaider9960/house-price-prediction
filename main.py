from sklearn.datasets import fetch_california_housing

housing= fetch_california_housing(as_frame=True)

print(housing.frame.head())

print("Dataset shape:", housing.frame.shape)
print("Column names:", housing.frame.columns.tolist())

print(housing.frame.isnull().sum())
print(housing.frame.describe())

X = housing.frame.drop("MedHouseVal", axis=1)

y = housing.frame["MedHouseVal"]

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")
y_pred = model.predict(X_test)
y_pred = model.predict(X_test)

print("Actual values:", y_test.iloc[:5].values)
print("Predicted values:", y_pred[:5])

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)

from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE:", rmse)
print("RMSE in dollars:", round(rmse * 100000, 2))
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)

import pandas as pd

comparison = pd.DataFrame({
    "Actual": y_test.values[:5] * 100000,
    "Predicted": y_pred[:5] * 100000
})

print(comparison.round(2))

import joblib

joblib.dump(model, "house_price_model.pkl")

print("Model saved successfully!")

import joblib
import os

joblib.dump(model, "house_price_model.pkl")

print("Model saved successfully!")
print("File location:", os.path.abspath("house_price_model.pkl"))
print("File exists:", os.path.exists("house_price_model.pkl"))

import joblib
from pathlib import Path

project_folder = Path(__file__).resolve().parent

model_path = project_folder / "house_price_model.pkl"

joblib.dump(model, model_path)

print("Model saved successfully!")
print("File location:", model_path)

