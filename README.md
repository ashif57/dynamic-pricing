<p>Project Overview
The goal of this project is to build a machine learning model to predict room prices dynamically based on various features. Dynamic pricing helps optimize revenue by adjusting prices in response to market demands, local events, competitor pricing, and other factors. The model uses XGBoost, a powerful gradient boosting algorithm, to achieve high accuracy and predictive power.

Dataset Description
The dataset includes the following features:

RoomPrice($): The target variable representing the room price in dollars.
LocalEvents/Attractions: Categorical feature indicating local events or attractions that may affect room demand.
LengthofStay(nights): Numerical feature indicating the number of nights a room is booked.
RoomAvailability: Numerical feature representing the number of available rooms.
RoomType: Categorical feature indicating the type of room (e.g., Pro, Standard).
Competitor1_Price, Competitor2_Price, Competitor3_Price, Competitor4_Price: Numerical features representing the prices of competitors.
Booking Time: Timestamp of the booking.
Steps Involved
Data Preprocessing:

Loading Data: Read the dataset from a CSV file.
Datetime Conversion: Convert the Booking Time column to datetime format and extract useful features such as day, month, and hour.
Categorical Encoding: Encode categorical features (RoomType and LocalEvents/Attractions) using one-hot encoding.
Handling Missing Values: Fill any missing values in the dataset.
Feature Scaling: Standardize numerical features to have zero mean and unit variance.
Model Training with Hyperparameter Tuning:

Splitting Data: Split the dataset into training and testing sets.
Defining Parameter Grid: Set up a grid of hyperparameters for the XGBoost model.
Grid Search Cross-Validation: Use GridSearchCV to find the best combination of hyperparameters based on cross-validation performance.
Training the Model: Train the XGBoost model using the best hyperparameters identified.
Model Evaluation:

Predictions: Make predictions on the test set.
Performance Metrics: Calculate and print the Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²) to evaluate model performance.
Prediction for New Data:

New Data Preparation: Prepare new data for prediction by ensuring it matches the format of the training data.
Prediction: Use the trained model to predict room prices for new data instances.
</p>
