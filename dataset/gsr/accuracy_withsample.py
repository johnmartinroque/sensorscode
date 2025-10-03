import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths for GSR data
high_path = "High_MWL"
low_path = "Low_MWL"

def load_and_prepare_data(high_path, low_path):
    X, y = [], []
    for i in range(2, 26):
        filename = f"p{i}h.csv"
        filepath = os.path.join(high_path, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, header=None)
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                if not df.empty:
                    features = extract_gsr_features(df)
                    X.append(features)
                    y.append(1)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    for i in range(2, 26):
        filename = f"p{i}l.csv"
        filepath = os.path.join(low_path, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, header=None)
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                if not df.empty:
                    features = extract_gsr_features(df)
                    X.append(features)
                    y.append(0)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    return np.array(X), np.array(y)

def extract_gsr_features(df):
    features = []
    for col in df.columns:
        column_data = df[col].dropna()
        if len(column_data) > 0:
            col_features = [
                np.mean(column_data),
                np.std(column_data),
                np.min(column_data),
                np.max(column_data),
                np.median(column_data),
                np.percentile(column_data, 25),
                np.percentile(column_data, 75),
                np.ptp(column_data),
            ]
            if len(column_data) > 1:
                differences = np.diff(column_data)
                col_features.extend([
                    np.mean(np.abs(differences)),
                    np.std(differences),
                    np.max(np.abs(differences)),
                ])
            else:
                col_features.extend([0, 0, 0])
            features.extend(col_features)
    if df.shape[1] > 1:
        try:
            correlation = df.corr().values[np.triu_indices(df.shape[1], k=1)]
            features.extend(correlation)
        except:
            pass
        column_means = df.mean()
        if len(column_means) > 1:
            mean_differences = [column_means[i] - column_means[j]
                                for i in range(len(column_means))
                                for j in range(i+1, len(column_means))]
            features.extend(mean_differences)
    return features

def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50))
    }

    results = {}
    print("Model Evaluation Results:")
    print("=" * 60)
    for name, model in models.items():
        try:
            if name in ['SVM', 'K-Nearest Neighbors', 'Neural Network', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X, y, cv=5)
            accuracy = accuracy_score(y_test, y_pred)
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()
            results[name] = {
                'accuracy': accuracy,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score,
                'cv_scores': cv_scores,
                'model': model,
                'scaler': scaler if name in ['SVM', 'K-Nearest Neighbors', 'Neural Network', 'Logistic Regression'] else None
            }
            print(f"\n{name}:")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Cross-validation: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")
        except Exception as e:
            print(f"\n{name} - Error: {e}")
            results[name] = None
    return results

def print_detailed_results(results, X, y):
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)
    valid_results = [(name, result) for name, result in results.items() if result is not None]
    if not valid_results:
        print("No valid models to evaluate.")
        return
    best_model_name, best_result = max(valid_results, key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {best_result['accuracy']:.4f}")
    best_model = best_result['model']
    scaler = best_result['scaler']
    X_train, X_test_new, y_train, y_test_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if scaler:
        best_model.fit(scaler.fit_transform(X_train), y_train)
        y_pred = best_model.predict(scaler.transform(X_test_new))
    else:
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test_new)
    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test_new, y_pred, target_names=['Low MWL', 'High MWL']))
    print(f"Confusion Matrix for {best_model_name}:")
    cm = confusion_matrix(y_test_new, y_pred)
    print(cm)

def predict_random_samples(results, X, y, n=5):
    print("\n" + "=" * 60)
    print(f"PREDICTIONS ON {n} RANDOM SAMPLES")
    print("=" * 60)

    # Pick random indices
    indices = np.random.choice(len(X), size=n, replace=False)
    sample_values = X[indices]
    sample_labels = y[indices]

    for i, (sample, true_label) in enumerate(zip(sample_values, sample_labels), 1):
        print(f"\nSample {i} (True: {'High MWL' if true_label == 1 else 'Low MWL'})")

        for name, result in results.items():
            if result is None:
                continue
            model = result['model']
            scaler = result['scaler']
            try:
                sample_input = sample.reshape(1, -1)  # make 2D
                if scaler:
                    pred = model.predict(scaler.transform(sample_input))[0]
                else:
                    pred = model.predict(sample_input)[0]

                pred_label = "High MWL" if pred == 1 else "Low MWL"
                print(f"  {name:<20}: {pred_label}")
            except Exception as e:
                print(f"  {name:<20}: Prediction error ({e})")

if __name__ == "__main__":
    print("Loading GSR data...")
    X, y = load_and_prepare_data(high_path, low_path)
    if len(X) == 0:
        print("No data loaded. Please check your file paths and data files.")
    else:
        print(f"\nData loaded successfully!")
        print(f"Total samples: {len(X)}")
        print(f"High MWL samples: {np.sum(y == 1)}")
        print(f"Low MWL samples: {np.sum(y == 0)}")
        print(f"Feature dimension: {X.shape[1]}")

        results = evaluate_models(X, y)
        print_detailed_results(results, X, y)

        # Predict on 5 random samples
        predict_random_samples(results, X, y, n=5)