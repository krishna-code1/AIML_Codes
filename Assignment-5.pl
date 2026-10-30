

# -------------------- STEP 1: IMPORT LIBRARIES --------------------

import time
import warnings
warnings.filterwarnings("ignore")

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
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    StackingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)

# In Google Colab, uncomment these lines if the CSV is not uploaded:
# from google.colab import files
# files.upload()

# -------------------- STEP 2: LOAD DATASET --------------------

df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())
print("\nDataset information:")
df.info()

# -------------------- STEP 3: EXPLORATORY DATA ANALYSIS --------------------

print("\n========== EDA SUMMARY ==========")
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nClass distribution:")
print(df["Class"].value_counts())
print("\nClass percentages:")
print((df["Class"].value_counts(normalize=True) * 100).round(4))
print("\nStatistical summary:")
print(df.describe())

genuine_count = int((df["Class"] == 0).sum())
fraud_count = int((df["Class"] == 1).sum())

print("\nGenuine transactions:", genuine_count)
print("Fraudulent transactions:", fraud_count)
print("Fraud percentage:", round(fraud_count / len(df) * 100, 4), "%")

# Graph 1: Original class distribution
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

# Graph 3: Amount by class
plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Amount", data=df)
plt.title("Transaction Amount by Class")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.show()

# -------------------- STEP 4: PREPROCESSING --------------------

# Remove rows with NaN values first
df = df.dropna().copy()

# Remove exact duplicate records before splitting.
df = df.drop_duplicates().copy()

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Fit scaler on training data only to prevent data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n========== PREPROCESSING ==========")
print("Dataset after duplicate and NaN removal:", df.shape)
print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)
print("\nTraining class distribution:")
print(y_train.value_counts())
print("\nTesting class distribution:")
print(y_test.value_counts())

# -------------------- STEP 5: HANDLE CLASS IMBALANCE --------------------

# Undersample only the training set. Keep all fraud cases and at most
# 10 genuine cases for each fraud case. The test set stays unchanged.
train_data = pd.DataFrame(X_train_scaled)
train_data["Class"] = y_train.to_numpy()

genuine_train = train_data[train_data["Class"] == 0]
fraud_train = train_data[train_data["Class"] == 1]

genuine_sample = resample(
    genuine_train,
    replace=False,
    n_samples=min(len(genuine_train), len(fraud_train) * 10),
    random_state=42
)

balanced_train = pd.concat([genuine_sample, fraud_train], axis=0)
balanced_train = balanced_train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_train_balanced = balanced_train.drop("Class", axis=1)
y_train_balanced = balanced_train["Class"].astype(int)

print("\n========== UNDERSAMPLING RESULTS ==========")
print(y_train_balanced.value_counts())
print("Balanced training shape:", X_train_balanced.shape)
print("Original test shape:", X_test_scaled.shape)

plt.figure(figsize=(7, 5))
sns.countplot(x=y_train_balanced)
plt.title("Training Class Distribution After Undersampling")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()

# -------------------- STEP 6: DEFINE BASE CLASSIFIERS --------------------

# Use the same base estimators for the individual models, stacking
# and voting so the comparison is consistent.
base_estimators = [
    ("svm", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    )),
    ("decision_tree", DecisionTreeClassifier(
        max_depth=None,
        class_weight="balanced",
        random_state=42
    )),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
    ("naive_bayes", GaussianNB())
]

base_models = {
    "SVM": base_estimators[0][1],
    "Decision Tree": base_estimators[1][1],
    "KNN": base_estimators[2][1],
    "Naive Bayes": base_estimators[3][1]
}

# -------------------- STEP 7: DEFINE ENSEMBLE CLASSIFIERS --------------------

# 7A. Stacking:
# Base classifiers generate out-of-fold predictions. Logistic Regression
# learns how to combine those predictions as the meta-classifier.
stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    cv=3,
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=False
)

# 7B. AdaBoost:
# A shallow decision tree is used as the weak learner.
# "estimator" is the current scikit-learn parameter name.
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

# 7C. Hard Voting:
# Each classifier casts one class vote; the majority class is selected.
voting_model = VotingClassifier(
    estimators=base_estimators,
    voting="hard",
    n_jobs=-1
)

all_models = {
    **base_models,
    "Stacking": stacking_model,
    "AdaBoost": adaboost_model,
    "Voting (Hard)": voting_model
}

# -------------------- STEP 8: TRAIN ALL MODELS --------------------

trained_models = {}
training_times = {}

print("\n========== MODEL TRAINING ==========")

for name, model in all_models.items():
    print("\nTraining:", name)
    start_time = time.time()

    model.fit(X_train_balanced, y_train_balanced)

    elapsed = time.time() - start_time
    trained_models[name] = model
    training_times[name] = elapsed

    print("Training completed.")
    print("Training time:", round(elapsed, 2), "seconds")

print("\nAll base and ensemble models trained successfully.")

# -------------------- STEP 9: EVALUATE ALL MODELS --------------------

results = []
predictions = {}
probabilities_for_loss = {}

for name, model in trained_models.items():

    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred

    # For models with predict_proba, use their positive-class probability.
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = None

    # Hard voting has no probability output by design.
    # For its Log Loss only, use the mean positive-class probability
    # from its fitted base classifiers. Hard-vote predictions are still
    # used for accuracy, precision, recall and F1-score.
    if name == "Voting (Hard)":
        base_probabilities = []
        for _, fitted_estimator in model.named_estimators_.items():
            base_probabilities.append(
                fitted_estimator.predict_proba(X_test_scaled)[:, 1]
            )
        y_prob = np.mean(base_probabilities, axis=0)
        y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)

    probabilities_for_loss[name] = y_prob

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    loss = log_loss(y_test, y_prob, labels=[0, 1])

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Log Loss": loss,
        "Training Time (sec)": training_times[name]
    })

    print("\n==========", name, "==========")
    print(classification_report(y_test, y_pred, zero_division=0))

results_df = pd.DataFrame(results)

print("\n========== BASE AND ENSEMBLE MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

# -------------------- STEP 10: CONFUSION MATRICES --------------------

for name, y_pred in predictions.items():
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

# -------------------- STEP 11: PERFORMANCE COMPARISON GRAPHS --------------------

# Accuracy
plt.figure(figsize=(11, 5))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.title("Accuracy Comparison: Base vs Ensemble Models")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Precision, Recall and F1-score
metric_data = results_df.melt(
    id_vars="Model",
    value_vars=["Precision", "Recall", "F1-Score"],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(12, 5))
sns.barplot(
    data=metric_data,
    x="Model",
    y="Score",
    hue="Metric"
)
plt.title("Precision, Recall and F1-Score Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Log Loss
plt.figure(figsize=(11, 5))
sns.barplot(data=results_df, x="Model", y="Log Loss")
plt.title("Log Loss Comparison (Lower Is Better)")
plt.xlabel("Model")
plt.ylabel("Log Loss")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# Training time
plt.figure(figsize=(11, 5))
sns.barplot(data=results_df, x="Model", y="Training Time (sec)")
plt.title("Training Time Comparison")
plt.xlabel("Model")
plt.ylabel("Training Time (seconds)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

# -------------------- STEP 12: COMPARATIVE FINDINGS --------------------

print("\n========== BEST MODEL BY EACH METRIC ==========")

best_accuracy = results_df.loc[results_df["Accuracy"].idxmax()]
best_precision = results_df.loc[results_df["Precision"].idxmax()]
best_recall = results_df.loc[results_df["Recall"].idxmax()]
best_f1 = results_df.loc[results_df["F1-Score"].idxmax()]
best_log_loss = results_df.loc[results_df["Log Loss"].idxmin()]

print("Highest Accuracy:", best_accuracy["Model"],
      round(best_accuracy["Accuracy"], 4))
print("Highest Precision:", best_precision["Model"],
      round(best_precision["Precision"], 4))
print("Highest Recall:", best_recall["Model"],
      round(best_recall["Recall"], 4))
print("Highest F1-Score:", best_f1["Model"],
      round(best_f1["F1-Score"], 4))
print("Lowest Log Loss:", best_log_loss["Model"],
      round(best_log_loss["Log Loss"], 4))

print("\n========== ENSEMBLE MODEL RESULTS ==========")
print(
    results_df[results_df["Model"].isin(
        ["Stacking", "AdaBoost", "Voting (Hard)"]
    )].round(4).to_string(index=False)
)

print("\n========== FINAL EXPERIMENT SUMMARY ==========")
print("Base classifiers: SVM, Decision Tree, KNN, Naive Bayes")
print("Ensemble classifiers: Stacking, AdaBoost, Hard Voting")
print("Evaluation metrics: Accuracy, Precision, Recall, F1-score, Log Loss")
print("The test set retained its original class distribution.")
print("Experiment completed successfully.")

# Optional: save the comparison table as a CSV file.
results_df.to_csv("ensemble_model_comparison.csv", index=False)
print("\nComparison table saved as ensemble_model_comparison.csv")
