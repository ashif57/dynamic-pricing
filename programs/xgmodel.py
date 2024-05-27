import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Step 1: Data Preprocessing
path = r'D:\faqbot\dynamic-pricing\data\modif.csv'

data = pd.read_csv(path)

# Convert 'Booking Time' column to datetime
data['Booking Time'] = pd.to_datetime(data['Booking Time'])

# Extract features from 'Booking Time'
data['Booking_Day'] = data['Booking Time'].dt.day
data['Booking_Month'] = data['Booking Time'].dt.month
data['Booking_Hour'] = data['Booking Time'].dt.hour

# Drop unnecessary columns
data = data.drop(['Booking Time'], axis=1)

# Encode categorical variables like 'RoomType' and 'LocalEvents/Attractions' using one-hot encoding
data = pd.get_dummies(data, columns=['RoomType', 'LocalEvents/Attractions'])

# Handling missing values if any
data.fillna(0, inplace=True)  # You may want to use more sophisticated methods for handling missing values

# Step 2: Feature Engineering (if needed)
# Normalizing numerical features
scaler = StandardScaler()
numerical_cols = ['LengthofStay(nights)', 'RoomAvailability', 'Competitor1_Price', 'Competitor2_Price', 'Competitor3_Price', 'Competitor4_Price']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Step 3: Model Training with Hyperparameter Tuning
X = data.drop('RoomPrice($)', axis=1)
y = data['RoomPrice($)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Create the XGBoost regressor
model = xgb.XGBRegressor()

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best R-squared:", best_score)

# Train XGBoost model with the best parameters
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = best_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

# Now, let's predict the room price for new data
# Example new data (replace with your own data)
new_data = {
    'LocalEvents/Attractions': ['Concert'],  # Example value for a categorical feature
    'LengthofStay(nights)': [3],             # Example value for a numerical feature
    'RoomAvailability': [2],                 # Example value for a numerical feature
    'RoomType': ['Pro'],                      # Example value for a categorical feature
    'Competitor1_Price': [130],               # Example value for a numerical feature
    'Competitor2_Price': [135],               # Example value for a numerical feature
    'Competitor3_Price': [125],               # Example value for a numerical feature
    'Competitor4_Price': [145],               # Example value for a numerical feature
    'Booking_Day': [18],                      # Example value for the day of booking
    'Booking_Month': [4],                     # Example value for the month of booking
    'Booking_Hour': [8]                       # Example value for the hour of booking
}

# Create a DataFrame from the new data
X_new = pd.DataFrame(new_data)

# Ensure the 'RoomType' and 'LocalEvents/Attractions' columns are one-hot encoded
X_new = pd.get_dummies(X_new, columns=['RoomType', 'LocalEvents/Attractions'])

# Align columns of X_new with training data
X_new_aligned = X_new.reindex(columns=X.columns, fill_value=0)

# Predict room price for the aligned new data
predicted_price = best_model.predict(X_new_aligned)
print("Predicted Room Price:", predicted_price)
