

# ============================================================
# STEP 1: IMPORT LIBRARIES AND LOAD DATASET
# ============================================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix
)

# If running in Google Colab and the CSV is not uploaded yet,
# uncomment the next two lines:
# from google.colab import files
# files.upload()

df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!")
print("\nFirst five records:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n========== EDA: MISSING VALUES ==========")
print("Total missing values:", df.isnull().sum().sum())

print("\n========== EDA: DUPLICATE RECORDS ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\n========== EDA: STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== EDA: CLASS DISTRIBUTION ==========")
print(df["Class"].value_counts())
print("\nClass percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

genuine_count = (df["Class"] == 0).sum()
fraud_count = (df["Class"] == 1).sum()

print("\nGenuine transactions:", genuine_count)
print("Fraudulent transactions:", fraud_count)
print("Fraud percentage:", round(fraud_count / len(df) * 100, 4), "%")

# Graph 1: Genuine vs fraud transactions
plt.figure(figsize=(7, 5))
sns.countplot(x="Class", data=df)
plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()

# Graph 2: Transaction amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Amount"], bins=50)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Graph 3: Transaction amount by class
plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Amount", data=df)
plt.title("Transaction Amount by Class")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.show()

# Graph 4: Correlation heatmap
plt.figure(figsize=(14, 10))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\n========== EDA SUMMARY ==========")
print("Total transactions:", len(df))
print("Genuine transactions:", genuine_count)
print("Fraudulent transactions:", fraud_count)
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())


# ============================================================
# STEP 3: DATA PREPROCESSING
# ============================================================

# Remove duplicate rows before splitting the data.
df = df.drop_duplicates().copy()
print("\nDataset shape after removing duplicates:", df.shape)

# Separate input features and target variable.
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Stratified split preserves the original class ratio in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Fit scaler only on training data to avoid data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\n========== PREPROCESSING SUMMARY ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])
print("Fraud cases in training:", int((y_train == 1).sum()))
print("Fraud cases in testing:", int((y_test == 1).sum()))


# ============================================================
# STEP 4: HANDLE CLASS IMBALANCE USING RANDOM UNDERSAMPLING
# ============================================================

# Undersampling is applied ONLY to the training set.
# The test set remains unchanged to represent the original distribution.

train_data = pd.DataFrame(X_train_scaled)
train_data["Class"] = y_train.to_numpy()

genuine_data = train_data[train_data["Class"] == 0]
fraud_data = train_data[train_data["Class"] == 1]

print("\n========== BEFORE UNDERSAMPLING ==========")
print("Genuine transactions:", len(genuine_data))
print("Fraud transactions:", len(fraud_data))

# Keep up to 10 genuine examples for each fraud example.
genuine_sample = resample(
    genuine_data,
    replace=False,
    n_samples=min(len(genuine_data), len(fraud_data) * 10),
    random_state=42
)

balanced_data = pd.concat([genuine_sample, fraud_data], axis=0)
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_balanced = balanced_data.drop("Class", axis=1)
y_train_balanced = balanced_data["Class"].astype(int)

print("\n========== AFTER UNDERSAMPLING ==========")
print(y_train_balanced.value_counts())
print("Balanced training shape:", X_train_balanced.shape)
print("Original test records (unchanged):", len(X_test_scaled))

plt.figure(figsize=(7, 5))
sns.countplot(x=y_train_balanced)
plt.title("Training Class Distribution After Undersampling")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()


# ============================================================
# STEP 5: BUILD AND TRAIN FOUR CLASSIFICATION MODELS
# ============================================================

models = {
    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),
    "Naive Bayes": GaussianNB()
}

trained_models = {}

print("\n========== MODEL TRAINING ==========")

for name, model in models.items():
    print("\nTraining:", name)
    start_time = time.time()

    model.fit(X_train_balanced, y_train_balanced)

    elapsed_time = time.time() - start_time
    trained_models[name] = model

    print("Training completed successfully!")
    print("Training time:", round(elapsed_time, 2), "seconds")

print("\nAll four models have been trained.")


# ============================================================
# STEP 6: EVALUATE MODELS USING REQUIRED METRICS
# ============================================================

results = []

print("\n========== MODEL EVALUATION ==========")

for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    loss = log_loss(y_test, y_prob, labels=[0, 1])

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Log Loss": loss
    })

    print("\n", name)
    print(classification_report(y_test, y_pred, zero_division=0))

results_df = pd.DataFrame(results)

print("\n========== FINAL PERFORMANCE COMPARISON ==========")
print(results_df.round(4).to_string(index=False))


# ============================================================
# STEP 7: CONFUSION MATRICES AND COMPARISON GRAPHS
# ============================================================

# Confusion matrix for each model
for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Fraud"],
        yticklabels=["Genuine", "Fraud"]
    )
    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.show()

# Accuracy comparison
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Algorithm", y="Accuracy")
plt.title("Accuracy Comparison of Classification Models")
plt.xlabel("Classification Algorithms")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Precision, Recall and F1-score comparison
metrics_df = results_df.melt(
    id_vars="Algorithm",
    value_vars=["Precision", "Recall", "F1-Score"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=metrics_df,
    x="Algorithm",
    y="Score",
    hue="Metric"
)
plt.title("Precision, Recall and F1-Score Comparison")
plt.xlabel("Classification Algorithms")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Log Loss comparison
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Algorithm", y="Log Loss")
plt.title("Log Loss Comparison of Classification Models")
plt.xlabel("Classification Algorithms")
plt.ylabel("Log Loss (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

print("\nAll confusion matrices and comparison graphs generated.")


# ============================================================
# STEP 8: COMPARATIVE ANALYSIS AND FINAL FINDINGS
# ============================================================

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

best_accuracy = results_df.loc[results_df["Accuracy"].idxmax()]
best_precision = results_df.loc[results_df["Precision"].idxmax()]
best_recall = results_df.loc[results_df["Recall"].idxmax()]
best_f1 = results_df.loc[results_df["F1-Score"].idxmax()]
best_loss = results_df.loc[results_df["Log Loss"].idxmin()]

print("\n========== BEST PERFORMING MODELS ==========")
print("Highest Accuracy:", best_accuracy["Algorithm"],
      "-", round(best_accuracy["Accuracy"], 4))
print("Highest Precision:", best_precision["Algorithm"],
      "-", round(best_precision["Precision"], 4))
print("Highest Recall:", best_recall["Algorithm"],
      "-", round(best_recall["Recall"], 4))
print("Highest F1-Score:", best_f1["Algorithm"],
      "-", round(best_f1["F1-Score"], 4))
print("Lowest Log Loss:", best_loss["Algorithm"],
      "-", round(best_loss["Log Loss"], 4))

print("\n========== COMPARATIVE ANALYSIS ==========")
for _, row in results_df.iterrows():
    print("\nAlgorithm:", row["Algorithm"])
    print("Accuracy:", round(row["Accuracy"], 4))
    print("Precision:", round(row["Precision"], 4))
    print("Recall:", round(row["Recall"], 4))
    print("F1-Score:", round(row["F1-Score"], 4))
    print("Log Loss:", round(row["Log Loss"], 4))

print("\n========== FINAL FINDINGS ==========")
print("Best model based on F1-score:", best_f1["Algorithm"])
print("Best model based on fraud recall:", best_recall["Algorithm"])
print("Best model based on Log Loss:", best_loss["Algorithm"])
print("\nClassification experiment completed successfully.")
