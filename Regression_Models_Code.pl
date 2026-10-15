import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_FILE = "/car_data[1].csv"
TARGET = "Selling_Price"
FEATURE = "Present_Price"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def evaluate_model(y_test, y_pred):
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2  :", r2_score(y_test, y_pred))


def plot_predictions(y_test, y_pred, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)

    low = min(np.min(y_test), np.min(y_pred))
    high = max(np.max(y_test), np.max(y_pred))

    plt.plot([low, high], [low, high], "r--")
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(title)
    plt.tight_layout()
    plt.show()


data = pd.read_csv(DATA_FILE)
data = data.dropna(subset=[TARGET]).copy()

X_simple = data[[FEATURE]].to_numpy(dtype=float)
y_simple = data[TARGET].to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X_simple,
    y_simple,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# ============================================================
# 1. SIMPLE LINEAR REGRESSION WITHOUT LIBRARY
# ============================================================

def estimate_coef(x, y):
    n = np.size(x)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    ss_xy = np.sum((x - mean_x) * (y - mean_y))
    ss_xx = np.sum((x - mean_x) ** 2)

    if np.isclose(ss_xx, 0):
        raise ValueError("Feature must contain different values.")

    b1 = ss_xy / ss_xx
    b0 = mean_y - b1 * mean_x

    return b0, b1


def plot_regression_line(x, y, b, title):
    y_pred = b[0] + b[1] * x

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="magenta", s=30)

    order = np.argsort(x)
    plt.plot(x[order], y_pred[order], color="green")

    plt.xlabel("Present Price")
    plt.ylabel("Selling Price")
    plt.title(title)
    plt.tight_layout()
    plt.show()


print("\n1. SIMPLE LINEAR REGRESSION WITHOUT LIBRARY")

b0, b1 = estimate_coef(
    X_train.ravel(),
    y_train
)

y_pred_manual = b0 + b1 * X_test.ravel()

print("Intercept:", b0)
print("Slope:", b1)

evaluate_model(y_test, y_pred_manual)

plot_regression_line(
    X_test.ravel(),
    y_pred_manual,
    (b0, b1),
    "Simple Linear Regression - Manual"
)


# ============================================================
# 2. SIMPLE LINEAR REGRESSION WITH LIBRARY
# ============================================================

print("\n2. SIMPLE LINEAR REGRESSION WITH SCIKIT-LEARN")

simple_model = LinearRegression()

simple_model.fit(X_train, y_train)

y_pred_simple = simple_model.predict(X_test)

print("Intercept:", simple_model.intercept_)
print("Coefficient:", simple_model.coef_[0])

evaluate_model(y_test, y_pred_simple)

plot_predictions(
    y_test,
    y_pred_simple,
    "Simple Linear Regression - Scikit-learn"
)


# ============================================================
# PREPARE FEATURES FOR MULTIPLE LINEAR REGRESSION
# ============================================================

feature_columns = [
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner"
]

X_multiple = pd.get_dummies(
    data[feature_columns],
    drop_first=True,
    dtype=float
)

y_multiple = data[TARGET].to_numpy(dtype=float)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multiple.to_numpy(dtype=float),
    y_multiple,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# ============================================================
# 3. MULTIPLE LINEAR REGRESSION WITHOUT LIBRARY
# ============================================================

print("\n3. MULTIPLE LINEAR REGRESSION WITHOUT LIBRARY")

X_train_bias = np.column_stack([
    np.ones(X_train_m.shape[0]),
    X_train_m
])

X_test_bias = np.column_stack([
    np.ones(X_test_m.shape[0]),
    X_test_m
])

coefficients = np.linalg.pinv(X_train_bias) @ y_train_m

y_pred_multiple_manual = X_test_bias @ coefficients

print("Intercept:", coefficients[0])

print("Coefficients:")

for name, value in zip(X_multiple.columns, coefficients[1:]):
    print(name, ":", value)

evaluate_model(y_test_m, y_pred_multiple_manual)

plot_predictions(
    y_test_m,
    y_pred_multiple_manual,
    "Multiple Linear Regression - Manual"
)


# ============================================================
# 4. MULTIPLE LINEAR REGRESSION WITH LIBRARY
# ============================================================

print("\n4. MULTIPLE LINEAR REGRESSION WITH SCIKIT-LEARN")

multiple_model = LinearRegression()

multiple_model.fit(X_train_m, y_train_m)

y_pred_multiple = multiple_model.predict(X_test_m)

print("Intercept:", multiple_model.intercept_)

print("Coefficients:")

for name, value in zip(
    X_multiple.columns,
    multiple_model.coef_
):
    print(name, ":", value)

evaluate_model(y_test_m, y_pred_multiple)

plot_predictions(
    y_test_m,
    y_pred_multiple,
    "Multiple Linear Regression - Scikit-learn"
)


# ============================================================
# 5. LINEAR REGRESSION WITH GRADIENT DESCENT
# ============================================================

print("\n5. LINEAR REGRESSION WITH GRADIENT DESCENT")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_m)
X_test_scaled = scaler.transform(X_test_m)

X_train_gd = np.column_stack([
    np.ones(X_train_scaled.shape[0]),
    X_train_scaled
])

X_test_gd = np.column_stack([
    np.ones(X_test_scaled.shape[0]),
    X_test_scaled
])

theta = np.zeros(X_train_gd.shape[1])

learning_rate = 0.03
epochs = 20000
n = len(y_train_m)

loss_history = []

for epoch in range(epochs):
    y_pred_train = X_train_gd @ theta

    errors = y_pred_train - y_train_m

    gradient = (2 / n) * (X_train_gd.T @ errors)

    theta = theta - learning_rate * gradient

    if epoch % 100 == 0:
        mse = np.mean(
            (X_train_gd @ theta - y_train_m) ** 2
        )
        loss_history.append(mse)

y_pred_gd = X_test_gd @ theta

print("Learning rate:", learning_rate)
print("Epochs:", epochs)
print("Intercept:", theta[0])

evaluate_model(y_test_m, y_pred_gd)

plot_predictions(
    y_test_m,
    y_pred_gd,
    "Linear Regression - Gradient Descent"
)

plt.figure(figsize=(7, 5))
plt.plot(
    np.arange(len(loss_history)) * 100,
    loss_history
)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Learning Curve")
plt.tight_layout()
plt.show()


# ============================================================
# 6. POLYNOMIAL REGRESSION WITHOUT LIBRARY
# ============================================================

print("\n6. POLYNOMIAL REGRESSION WITHOUT LIBRARY")

degree = 2

X_train_poly = np.column_stack([
    X_train.ravel() ** power
    for power in range(degree + 1)
])

X_test_poly = np.column_stack([
    X_test.ravel() ** power
    for power in range(degree + 1)
])

poly_coefficients = (
    np.linalg.pinv(X_train_poly) @ y_train
)

y_pred_poly_manual = X_test_poly @ poly_coefficients

print("Polynomial degree:", degree)
print("Coefficients:", poly_coefficients)

evaluate_model(y_test, y_pred_poly_manual)

plot_predictions(
    y_test,
    y_pred_poly_manual,
    "Polynomial Regression - Manual"
)

x_curve = np.linspace(
    data[FEATURE].min(),
    data[FEATURE].max(),
    300
)

X_curve_poly = np.column_stack([
    x_curve ** power
    for power in range(degree + 1)
])

y_curve_manual = X_curve_poly @ poly_coefficients

plt.figure(figsize=(7, 5))
plt.scatter(
    data[FEATURE],
    data[TARGET],
    alpha=0.5,
    label="Actual Data"
)
plt.plot(
    x_curve,
    y_curve_manual,
    color="red",
    label="Polynomial Curve"
)
plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.title("Polynomial Regression - Manual")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. POLYNOMIAL REGRESSION WITH LIBRARY
# ============================================================

print("\n7. POLYNOMIAL REGRESSION WITH SCIKIT-LEARN")

polynomial_model = make_pipeline(
    PolynomialFeatures(
        degree=degree,
        include_bias=False
    ),
    LinearRegression()
)

polynomial_model.fit(X_train, y_train)

y_pred_poly = polynomial_model.predict(X_test)

evaluate_model(y_test, y_pred_poly)

plot_predictions(
    y_test,
    y_pred_poly,
    "Polynomial Regression - Scikit-learn"
)

y_curve_library = polynomial_model.predict(
    x_curve.reshape(-1, 1)
)

plt.figure(figsize=(7, 5))
plt.scatter(
    data[FEATURE],
    data[TARGET],
    alpha=0.5,
    label="Actual Data"
)
plt.plot(
    x_curve,
    y_curve_library,
    color="red",
    label="Polynomial Curve"
)
plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.title("Polynomial Regression - Scikit-learn")
plt.legend()
plt.tight_layout()
plt.show()
