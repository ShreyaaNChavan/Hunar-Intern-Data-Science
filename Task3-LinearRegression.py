# ===============================
# House Price Prediction Project
# Linear Regression
# ===============================

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load Dataset
df = pd.read_csv(r"C:\Users\admin\Desktop\Hunar-Intern-Data-Sci\dataset\house price data.csv")

print("Initial Dataset Shape:", df.shape)
print(df.head())

# 3. Data Cleaning

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Drop non-numeric / irrelevant columns
df.drop(columns=['street', 'city', 'statezip', 'country', 'date'], inplace=True)

# Check for null values
print("\nNull values before cleaning:\n", df.isnull().sum())

# Remove null values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("\nDataset Shape After Cleaning:", df.shape)

# 4. Feature Selection
X = df.drop('price', axis=1)   # Independent variables
y = df['price']                # Target variable

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# 6. Implement Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# 9. Test Model with New Data
# Example: [bedrooms, bathrooms, sqft_living, sqft_lot, floors,
# waterfront, view, condition, sqft_above, sqft_basement,
# yr_built, yr_renovated]

new_house = pd.DataFrame([[
    3, 2, 1800, 7500, 1,
    0, 0, 3, 1800, 0,
    1995, 0
]], columns=X.columns)

predicted_price = model.predict(new_house)

print("\nPredicted House Price:", predicted_price[0])
