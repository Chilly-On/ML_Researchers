import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Import all machine learning models
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

def load_data(filepath):
    """Load processed data."""
    return pd.read_csv(filepath)

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def tune_hyperparameters(X_train, y_train):
    """
    Tune hyperparameters for each model using GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        dict: Best-tuned models.
    """
    models = {
        # --- Linear Models ---
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        "Ridge Classifier": {
            "model": RidgeClassifier(),
            "params": {
                "alpha": [0.1, 1.0, 10.0]
            }
        },
        
        # --- Tree-Based Models ---
        "Decision Tree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(),
            "params": {
                "n_estimators": [100],
                "learning_rate": [0.01, 0.1]
            }
        },
        
        # --- Ensemble Methods ---
        "AdaBoost": {
            "model": AdaBoostClassifier(),
            "params": {
                "n_estimators": [50, 100]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "params": {
                "n_estimators": [100],
                "learning_rate": [0.01, 0.1]
            }
        },
        "LightGBM": {
            "model": LGBMClassifier(),
            "params": {
                "n_estimators": [100],
                "learning_rate": [0.01, 0.1]
            }
        },
        "CatBoost": {
            "model": CatBoostClassifier(verbose=0),
            "params": {
                "iterations": [100, 500]
            }
        },
        
        # --- Support Vector Machines ---
        "SVM": {
            "model": SVC(),
            "params": {
                "C": [0.1, 1],
                "kernel": ["rbf"]
            }
        },
        
        # --- Nearest Neighbors ---
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7]
            }
        },
        
        # --- Probabilistic Models ---
        "Gaussian Naive Bayes": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": [1e-9, 1e-7]
            }
        },
        
        # --- Discriminant Analysis ---
        "Linear Discriminant Analysis": {
            "model": LinearDiscriminantAnalysis(),
            "params": {
                "solver": ["svd", "lsqr"]
            }
        },
        "Quadratic Discriminant Analysis": {
            "model": QuadraticDiscriminantAnalysis(),
            "params": {
                "reg_param": [0.1]
            }
        }
    }

    best_models = {}
    for name, config in models.items():
        print(f"\nTuning {name}...")
        grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Save the best model
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Validation accuracy for {name}: {grid_search.best_score_}")

    return best_models

def train_models(X_train, y_train):
    """Train multiple machine learning models."""
    models = {
        # Linear Models
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Ridge Classifier": RidgeClassifier(),
        
        # Tree-Based Models
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        
        # Ensemble Methods
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0),
        
        # Support Vector Machines
        "SVM": SVC(),
        
        # Nearest Neighbors
        "K-Nearest Neighbors": KNeighborsClassifier(),
        
        # Probabilistic Models
        "Gaussian Naive Bayes": GaussianNB(),
        
        # Discriminant Analysis
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis()
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return metrics."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }
    return results

def save_models(models, directory="models/trained/"):
    """Save trained models to disk."""
    for name, model in models.items():
        filename = f"{directory}{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        print(f"Saved {name} to {filename}")

