import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the Dataset
path = r'D:\faqbot\dynamic-pricing\data\cleaned.csv'
df = pd.read_csv(path)

# Encode Categorical Features
encoder = OneHotEncoder(drop='first')  # Drop the first category to avoid multicollinearity
cat_features = ['Booking Time', 'Competitors Rates ($)', 'Local Events/Attractions', 'Room Type','Length of Stay (nights)','Room Availability']
encoded_df = pd.get_dummies(df, columns=cat_features, drop_first=True)

# Prepare Data
X = encoded_df.drop('Room Price ($)', axis=1)
y = encoded_df['Room Price ($)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost Model
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_reg.fit(X_train, y_train)

# Make Predictions
y_pred = xgb_reg.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print Evaluation Metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)
