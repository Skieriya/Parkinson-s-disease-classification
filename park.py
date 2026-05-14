import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold



df = pd.read_csv("dataset.csv", skiprows=1)

remove_cols = [
    "tqwt_kurtosisValue_dec_28", "tqwt_entropy_log_dec_31",
    "app_LT_TKEO_mean_3_coef", "std_9th_delta_delta", 
    "std_7th_delta", "tqwt_kurtosisValue_dec_20",
    "tqwt_TKEO_std_dec_11", "tqwt_entropy_log_dec_1",
    "tqwt_energy_dec_6", "mean_7th_delta_delta",
    "tqwt_medianValue_dec_1"
]

import pandas as pd

importances = model.get_feature_importance()

feature_importance = pd.Series(importances, index=df.columns)

low_importance_features = feature_importance[feature_importance <= 6].index

df_cleaned = df.drop(columns=low_importance_features)

print(f"Dropped {len(low_importance_features)} features.")
print(f"Remaining features: {df_cleaned.shape[1]}")

X = df.drop(columns=["class"] + remove_cols)
y = df["class"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf"]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

svm_model = GridSearchCV(
    SVC(),
    param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=2
)

svm_model.fit(X_train, y_train)

print("\nBest parameters:", svm_model.best_params_)


y_pred = svm_model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))





