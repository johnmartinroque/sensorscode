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

# Paths
high_path = "High_MWL"
low_path = "Low_MWL"

def load_and_prepare_data(high_path, low_path):
    """
    Load data from High_MWL and Low_MWL folders and prepare features/labels
    """
    X, y = [], []
    
    # Load High MWL files (p2h.csv to p25h.csv)
    for i in range(2, 26):  # p2 to p25
        filename = f"p{i}h.csv"
        filepath = os.path.join(high_path, filename)
        
        if os.path.exists(filepath):
            try:
                # Read CSV, skip header rows if they contain text
                df = pd.read_csv(filepath, header=None)
                
                # Remove non-numeric rows (like "Trial 3:3back,Trial 5:3back")
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                
                if not df.empty:
                    # Flatten the data to create features
                    features = df.values.flatten()
                    X.append(features)
                    y.append(1)  # High MWL = 1
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    # Load Low MWL files (p2l.csv to p25l.csv)
    for i in range(2, 26):
        filename = f"p{i}l.csv"
        filepath = os.path.join(low_path, filename)
        
        if os.path.exists(filepath):
            try:
                # Read CSV, skip header rows if they contain text
                df = pd.read_csv(filepath, header=None)
                
                # Remove non-numeric rows
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                
                if not df.empty:
                    # Flatten the data to create features
                    features = df.values.flatten()
                    X.append(features)
                    y.append(0)  # Low MWL = 0
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return np.array(X), np.array(y)

def evaluate_models(X, y):
    """
    Evaluate multiple machine learning models
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(
    random_state=42,
    n_estimators=50,           # Reduce from 100
    max_depth=3,               # Limit tree depth
    learning_rate=0.05,        # Lower learning rate
    subsample=0.8,             # Stochastic gradient boosting
    min_samples_split=20,      # Require more samples to split
    min_samples_leaf=10,       # Require more samples per leaf
    validation_fraction=0.1,   # Early stopping
    n_iter_no_change=5         # Stop if no improvement
),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    print("Model Evaluation Results:")
    print("=" * 60)
    
    for name, model in models.items():
        try:
            # Train and predict
            if name in ['SVM', 'K-Nearest Neighbors', 'Neural Network']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Cross-validation with scaled data
                cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()
            
            results[name] = {
                'accuracy': accuracy,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score,
                'cv_scores': cv_scores
            }
            
            print(f"\n{name}:")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Cross-validation: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")
            
        except Exception as e:
            print(f"\n{name} - Error: {e}")
            results[name] = None
    
    return results, X_train, X_test, y_train, y_test, models

def print_detailed_results(results, X_test, y_test, models, X, y):
    """
    Print detailed results for the best performing model
    """
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)
    
    # Find best model based on test accuracy
    best_model_name = max(
        [(name, result['accuracy']) for name, result in results.items() if result is not None],
        key=lambda x: x[1]
    )[0]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Print classification report for best model
    best_model = models[best_model_name]
    
    # Create a new train/test split for consistency
    X_train, X_test_new, y_train, y_test_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale if needed
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_new)
    
    if best_model_name in ['SVM', 'K-Nearest Neighbors', 'Neural Network']:
        best_model.fit(X_train_scaled, y_train)
        y_pred = best_model.predict(X_test_scaled)
    else:
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test_new)
    
    print(f"\nClassification Report for {best_model_name}:")
    print(classification_report(y_test_new, y_pred, target_names=['Low MWL', 'High MWL']))
    
    print(f"Confusion Matrix for {best_model_name}:")
    print(confusion_matrix(y_test_new, y_pred))

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    X, y = load_and_prepare_data(high_path, low_path)
    
    print(f"Loaded {len(X)} samples")
    print(f"High MWL samples: {np.sum(y == 1)}")
    print(f"Low MWL samples: {np.sum(y == 0)}")
    print(f"Feature dimension: {X.shape[1] if len(X) > 0 else 0}")
    
    if len(X) == 0:
        print("No data loaded. Please check your file paths and naming convention.")
    else:
        # Evaluate models
        results, X_train, X_test, y_train, y_test, models = evaluate_models(X, y)
        
        # Print detailed results
        print_detailed_results(results, X_test, y_test, models, X, y)

        
        # Summary table
        print("\n" + "=" * 60)
        print("SUMMARY TABLE")
        print("=" * 60)
        print(f"{'Model':<25} {'Test Accuracy':<15} {'CV Mean':<10} {'CV Std':<10}")
        print("-" * 60)
        
        for name, result in results.items():
            if result is not None:
                print(f"{name:<25} {result['accuracy']:<15.4f} {result['mean_cv_score']:<10.4f} {result['std_cv_score']:<10.4f}")