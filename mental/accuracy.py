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

# Path to your dataset
dataset_path = "mental_health_wearable_data.csv"  # change to your file name

def load_and_prepare_data(dataset_path):
    """
    Load dataset and extract only GSR_Values + Target
    """
    df = pd.read_csv(dataset_path)

    # Features: GSR_Values (reshape into 2D)
    X = df["GSR_Values"].values.reshape(-1, 1)

    # Labels: Target
    y = df["Target"].values

    return X, y

def evaluate_models(X, y):
    """
    Evaluate multiple machine learning models without saving them
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }

    results = {}

    print("Model Evaluation Results:")
    print("=" * 60)

    for name, model in models.items():
        try:
            # Always use scaled features (since GSR is single-valued)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)

            accuracy = accuracy_score(y_test, y_pred)
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()

            results[name] = {
                'accuracy': accuracy,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score,
                'cv_scores': cv_scores,
                'model': model
            }

            print(f"\n{name}:")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Cross-validation: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")

        except Exception as e:
            print(f"\n{name} - Error: {e}")
            results[name] = None

    return results, X_train_scaled, X_test_scaled, y_train, y_test, models, scaler

def print_detailed_results(results, X_test, y_test, models, X, y, scaler):
    """
    Print detailed results for the best performing model
    """
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

    best_model = models[best_model_name]

    X_train, X_test_new, y_train, y_test_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_new)

    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test_new, y_pred, target_names=['Class 0', 'Class 1']))

    print(f"Confusion Matrix for {best_model_name}:")
    print(confusion_matrix(y_test_new, y_pred))

if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_and_prepare_data(dataset_path)

    print(f"Samples: {len(X)}")
    print(f"Class 0 samples: {np.sum(y == 0)}")
    print(f"Class 1 samples: {np.sum(y == 1)}")

    if len(X) == 0:
        print("No data found!")
    else:
        results, X_train, X_test, y_train, y_test, models, scaler = evaluate_models(X, y)
        print_detailed_results(results, X_test, y_test, models, X, y, scaler)

        print("\n" + "=" * 60)
        print("SUMMARY TABLE")
        print("=" * 60)
        print(f"{'Model':<25} {'Test Accuracy':<15} {'CV Mean':<10} {'CV Std':<10}")
        print("-" * 60)

        for name, result in results.items():
            if result is not None:
                print(f"{name:<25} {result['accuracy']:<15.4f} {result['mean_cv_score']:<10.4f} {result['std_cv_score']:<10.4f}")

        print("\nAll models were evaluated using only GSR values.")
