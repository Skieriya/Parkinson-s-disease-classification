import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("dataset.csv", skiprows=1)

X = df.drop(columns=["class"])
y = df["class"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



def clip_outliers(df, lower=0.01, upper=0.99):
    quantiles_low = df.quantile(lower)
    quantiles_high = df.quantile(upper)
    return df.clip(quantiles_low, quantiles_high, axis=1)

X_train = clip_outliers(X_train)



X_train, y_train = SMOTE().fit_resample(X_train, y_train)



experiment_name = "Parkinsons_Classification_v5"
mlflow.set_experiment(experiment_name)



def train_and_log(model, name):
    with mlflow.start_run(run_name=name) as run:

        params = model.get_params()
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} - Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        cm_path = f"confusion_matrix_{name}.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()

        mlflow.log_artifact(cm_path)

        # Classification Report
        plt.figure(figsize=(8, 4))
        plt.text(0.01, 0.05, classification_report(y_test, y_pred), fontsize=12)
        plt.axis("off")
        cr_path = f"classification_report_{name}.png"
        plt.savefig(cr_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cr_path)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        return run.info.run_id


models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1),
    "XGBoost": XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=300),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, verbosity=-1),
    "CatBoost": CatBoostClassifier(verbose=0, iterations=300, learning_rate=0.05),

}


for name, model in models.items():
    print(f"Training {name}")
    train_and_log(model, name)


experiment = mlflow.get_experiment_by_name(experiment_name)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1 DESC"],
    max_results=1
)

if not runs.empty:
    best_run_id = runs.iloc[0].run_id
    best_f1 = runs.iloc[0]["metrics.f1"]
    print(f"\nBest Model: {runs.iloc[0]['tags.mlflow.runName']} with F1: {best_f1:.4f}")

    model_uri = f"runs:/{best_run_id}/model"
    model_name = "Best_Classifier_Model"

    mlflow.register_model(model_uri, model_name)
    print(f"Model registered as: {model_name}")

    prod = mlflow.sklearn.load_model(f"models:/{model_name}/1")

importances = prod.feature_importances_

try:
    feature_names = list(X.columns)
except:
    feature_names = getattr(prod, "feature_names_", None)

if feature_names is None:
    feature_names = [f"feature_{i}" for i in range(len(importances))]


min_len = min(len(feature_names), len(importances))
feature_names = feature_names[:min_len]
importances = importances[:min_len]

feature_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 10 Features:")
print(feature_df.head(10))

feature_df.to_csv("all_feature_importances.csv", index=False)