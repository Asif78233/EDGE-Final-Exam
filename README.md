# EDGE-Final-Exam


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

oil_data = pd.read_csv(r'E:\Asif\PetroleumPrices.csv')


p_oil_data['Date'] = pd.to_datetime(p_oil_data['Date'], format='%d-%b-%y', errors='coerce')
p_oil_data['Year'] = p_oil_data['Date'].dt.year=
p_oil_data = p_oil_data.groupby('Year')['Price'].mean().reset_index() 


X = p_oil_data[['Year']]
y = p_oil_data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error of the model: {mse:.2f}")
print(f"Model Accuracy (R^2): {r2 * 100:.2f}%")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, model.predict(X), color='red', label='Predicted Trend')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.title('Oil Price Trend and Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Predict for a manually input year
manual_year = int(input("Enter a year to predict petroleum price: "))
predicted_price = model.predict([[manual_year]])[0]
print(f"Predicted petroleum price for the year {manual_year}: ${predicted_price:.2f}")
