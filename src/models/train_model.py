from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os #Here
import pandas as pd
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
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler #Here
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def load_data(filepath):
    """Load processed data."""
    return pd.read_csv(filepath)

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def tune_hyperparameters(X_train, y_train, X_val, y_val):
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
    
    param_grid = {
        "n_components": [2, 3],
        "covariance_type": ["diag"],
        "n_iter": [100, 200]
    }

    best_hmm_models = tune_hmm(X_train.to_numpy(), y_train.to_numpy(), X_val.to_numpy(), y_val.to_numpy(), param_grid)
    best_models["HMM"] = best_hmm_models

    # hmm = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200)
    # hmm.fit(X_train)
    # best_models["HMM"] = hmm
    return best_models

# Define a simple MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_MLP(X_train, y_train, X_val, y_val, X_test, y_test, input_size, epochs=100, batch_size=32, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert pandas DataFrames/Series to numpy if needed
    if hasattr(X_train, 'values'):
        X_train = X_train.values
        X_val = X_val.values
        X_test = X_test.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
        y_val = y_val.values
        y_test = y_test.values

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)


    model = MLP(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation loss
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
                correct += ((pred > 0.5) == yb).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Final test evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            correct += ((pred > 0.5) == yb).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
    return model

def train_reduced_dimension_per_class(X_train, y_train, model_class, pca_components=2, **model_kwargs):
    """
    Train a separate model per class with PCA-reduced features.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_class: scikit-learn model class.
        pca_components (int): Number of PCA components.
        model_kwargs: Additional keyword arguments for the model.

    Returns:
        dict: Dictionary of trained models per class and PCA transformer.
    """
    class_models = {}
    pca = PCA(n_components=pca_components)
    X_reduced = pca.fit_transform(X_train)

    for label in np.unique(y_train):
        model = model_class(**model_kwargs)
        X_class = X_reduced[y_train == label]
        y_class = y_train[y_train == label]
        model.fit(X_class, y_class)
        class_models[label] = model

    return {"models": class_models, "pca": pca}
def predict_reduced_dimension(per_class_model_dict, X_test):
    """
    Predict using per-class reduced dimension models.
    Select the model with the highest confidence.

    Args:
        per_class_model_dict (dict): Contains 'models' and 'pca'.
        X_test (pd.DataFrame): Test data.

    Returns:
        list: Predicted labels.
    """
    models = per_class_model_dict["models"]
    pca = per_class_model_dict["pca"]
    X_reduced = pca.transform(X_test)
    predictions = []

    for x in X_reduced:
        scores = {}
        for label, model in models.items():
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba([x])[0].max()
                    scores[label] = prob
                else:
                    pred = model.predict([x])[0]
                    scores[label] = 1.0 if pred == label else 0.0
            except:
                scores[label] = float('-inf')

        best_label = max(scores, key=scores.get)
        predictions.append(best_label)

    return predictions

def tune_reduced_dimension(X_train, y_train, X_val, y_val, model_class, param_grid, pca_components=2):
    """
    Tune model on PCA-reduced features with GridSearchCV.
    """
    scaler = StandardScaler()
    pca = PCA(n_components=pca_components)
    pipeline = Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("clf", model_class())
    ])

    grid = GridSearchCV(pipeline, {"clf__" + k: v for k, v in param_grid.items()}, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    print(f"Best parameters: {grid.best_params_}")
    print(f"Validation accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def train_hmm_per_class(X_train, y_train, n_components=3, n_iter=100):
    """
    Train one GaussianHMM per class on PCA-reduced features.
    """
    models = {}
    print("Reducing dimensionality with PCA for HMM input...")
    pca = PCA(n_components=2)  # Keep it low to avoid huge parameter count
    X_train_reduced = pca.fit_transform(X_train)

    for label in np.unique(y_train):
        print(f"Training HMM for class {label}...")
        X_class = X_train_reduced[y_train == label]

        # Treat each row as a 1-step sequence
        model = GaussianHMM(n_components=n_components, n_iter=n_iter,
                            covariance_type="diag", random_state=42)

        # Fit as a single concatenated sequence, with lengths telling how many "sequences"
        lengths = [1] * len(X_class)
        model.fit(X_class, lengths)
        models[label] = model

    return models

def predict_hmm(models, X_test):
    """
    Predict classes using trained HMMs by selecting the model with the highest likelihood.

    Args:
        models (dict): A dictionary where keys are class labels and values are trained GaussianHMMs.
        X_test (np.ndarray or pd.DataFrame): Test data of shape (n_samples, n_features).

    Returns:
        list: Predicted class labels.
    """
    X = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test
    predictions = []

    for sample in X:
        sample = sample.reshape(1, -1)  # ensure 2D shape
        scores = {}

        for label, model in models.items():
            try:
                score = model.score(sample)
                scores[label] = score
            except:
                scores[label] = float('-inf')  # fallback if model can't score

        best_label = max(scores, key=scores.get)
        predictions.append(best_label)

    return predictions

def tune_hmm(X_train, y_train, X_val, y_val, param_grid):
    best_score = float('-inf')
    best_model = None

    for n_components in param_grid["n_components"]:
        for covariance_type in param_grid["covariance_type"]:
            for n_iter in param_grid["n_iter"]:
                try:
                    # Fit one HMM per class
                    models = {}
                    classes = np.unique(y_train)
                    for label in classes:
                        model = GaussianHMM(n_components=n_components,
                                            covariance_type=covariance_type,
                                            n_iter=n_iter)
                        X_class = X_train[y_train == label]
                        model.fit(X_class)
                        models[label] = model
                    
                    # Predict by max likelihood
                    preds = []
                    for x in X_val:
                        scores = {label: model.score([x]) for label, model in models.items()}
                        preds.append(max(scores, key=scores.get))

                    acc = accuracy_score(y_val, preds)

                    if acc > best_score:
                        best_score = acc
                        best_model = models
                        print(f"New best model: acc={acc:.4f}, n_components={n_components}, "
                              f"cov_type={covariance_type}, n_iter={n_iter}")

                except Exception as e:
                    print(f"Failed with n_components={n_components}, cov_type={covariance_type}, "
                          f"n_iter={n_iter} due to {e}")
                    continue

    return best_model

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

    # Train HMM models per class
    print("Training Hidden Markov Model (HMM)...")
    hmm_models = train_hmm_per_class(X_train, y_train)
    trained_models["HMM"] = hmm_models

    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and return metrics."""
    results = {}
    for name, model in models.items():
        if name == "HMM":
            y_pred = predict_hmm(model, X_test)
        else:
            y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    return results

def save_models(models, directory="models/trained/"):
    """Save trained models to disk."""
    for name, model in models.items():
        filename = f"{directory}{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
    print(f"Saved {name} to {filename}")