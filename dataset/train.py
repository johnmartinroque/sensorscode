import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Paths
high_path = "ppg/High_MWL"
low_path = "ppg/Low_MWL"

X, y = [], []

def extract_features(df):
    features = []
    for col in df.columns:
        col_data = df[col].dropna().values
        if len(col_data) == 0:  # skip empty columns
            continue
        features += [
            np.mean(col_data),
            np.std(col_data),
            np.min(col_data),
            np.max(col_data),
            np.median(col_data),
        ]
    return features

# High workload = 1
for file in os.listdir(high_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(high_path, file))
        X.append(extract_features(df))
        y.append(1)

# Low workload = 0
for file in os.listdir(low_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(low_path, file))
        X.append(extract_features(df))
        y.append(0)

X = np.array(X)
y = np.array(y)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.3f}")
