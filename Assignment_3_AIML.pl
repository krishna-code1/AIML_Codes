#1. Develop Regression Models for any real world application and Evaluate the performance using appropriate metrics. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("/final_dataset.csv")

# Remove unnecessary column
df = df.drop("Unnamed: 0", axis=1)

# Remove missing values
df = df.dropna()

# Features and target
X = df[["beds", "baths", "size"]]
y = df["price"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", len(X_train))
print("Testing data :", len(X_test))

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nPerformance:")
print("MAE      :", round(mae, 2))
print("MSE      :", round(mse, 2))
print("RMSE     :", round(rmse, 2))
print("R2 Score :", round(r2, 4))

# Show actual and predicted values
print("\nActual vs Predicted:")

for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print("Actual:", round(actual, 2),
          "Predicted:", round(predicted, 2))

# -------------------------------
# Regression Line
# -------------------------------

# Use house size for visualization
x = df[["size"]]
price = df["price"]

# Split size and price for graph
x_train, x_test, price_train, price_test = train_test_split(
    x, price, test_size=0.2, random_state=42
)

graph_model = LinearRegression()
graph_model.fit(x_train, price_train)

# Prediction line
x_line = np.linspace(
    x["size"].min(),
    x["size"].max(),
    100
).reshape(-1, 1)

y_line = graph_model.predict(x_line)

# Plot
plt.figure(figsize=(8, 5))

plt.scatter(
    x_train,
    price_train,
    label="Training Data",
    alpha=0.5
)

plt.scatter(
    x_test,
    price_test,
    label="Testing Data",
    alpha=0.5
)

plt.plot(
    x_line,
    y_line,
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Price Regression")
plt.legend()
plt.grid(True)

plt.show()

 #2. Implement and Analyze the performance of Gradient Descent optimization for Linear Regression.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("/final_dataset.csv")

# Remove unnecessary column
df = df.drop("Unnamed: 0", axis=1)

# Remove missing values
df = df.dropna(subset=["size", "price"])

# Input and target
X = df[["size"]]
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", len(X_train))
print("Testing data :", len(X_test))

# Convert to numpy
x_train = X_train["size"].values
x_test = X_test["size"].values

y_train = y_train.values
y_test = y_test.values

# Scale training data
mean = np.mean(x_train)
std = np.std(x_train)

x_train_scaled = (x_train - mean) / std
x_test_scaled = (x_test - mean) / std

# Initial values
m = 0
b = 0

learning_rate = 0.01
epochs = 1000

losses = []

# Gradient Descent
for i in range(epochs):

    # Prediction using training data
    y_pred = m * x_train_scaled + b

    # Error
    error = y_pred - y_train

    # Loss
    loss = np.mean(error ** 2)
    losses.append(loss)

    # Gradients
    dm = (2 / len(x_train_scaled)) * np.sum(
        error * x_train_scaled
    )

    db = (2 / len(x_train_scaled)) * np.sum(error)

    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

# -------------------------------
# Test the trained model
# -------------------------------

test_pred = m * x_test_scaled + b

# Performance
mae = mean_absolute_error(y_test, test_pred)
mse = mean_squared_error(y_test, test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, test_pred)

print("\nGradient Descent Results")
print("-----------------------")

print("Slope     :", round(m, 2))
print("Intercept :", round(b, 2))

print("\nTest Performance:")
print("MAE       :", round(mae, 2))
print("MSE       :", round(mse, 2))
print("RMSE      :", round(rmse, 2))
print("R2 Score  :", round(r2, 4))

# Actual vs predicted
print("\nActual vs Predicted:")

for actual, predicted in zip(y_test[:10], test_pred[:10]):
    print("Actual:", round(actual, 2),
          "Predicted:", round(predicted, 2))

# -------------------------------
# Regression Line
# -------------------------------

x_line = np.linspace(
    X["size"].min(),
    X["size"].max(),
    100
)

x_line_scaled = (x_line - mean) / std

y_line = m * x_line_scaled + b

plt.figure(figsize=(8, 5))

plt.scatter(
    x_train,
    y_train,
    label="Training Data",
    alpha=0.5
)

plt.scatter(
    x_test,
    y_test,
    label="Testing Data",
    alpha=0.5
)

plt.plot(
    x_line,
    y_line,
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid(True)

plt.show()

# -------------------------------
# Loss Graph
# -------------------------------

plt.figure(figsize=(8, 5))

plt.plot(losses)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Loss")

plt.grid(True)

plt.show()
 
