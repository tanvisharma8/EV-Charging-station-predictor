# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:09:07 2025

@author: Prisha D
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load cleaned dataset
df = pd.read_csv("cleaned_ev_data.csv")

# Features & target
X = df.drop('cost_per_unit', axis=1)
y = df['cost_per_unit']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Model Training Completed ✔")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save model
joblib.dump(model, "ev_cost_predictor.pkl")
print("Model saved as ev_cost_predictor.pkl ✔")
