import joblib
import pandas as pd

def load_model(model_path):
    """Load a trained model from disk."""
    return joblib.load(model_path)

def predict(model, data):
    """Make predictions using the trained model."""
    return model.predict(data)

if __name__ == "__main__":
    # Example usage
    model = load_model("models/trained/logistic_regression.pkl")
    new_data = pd.read_csv("data/processed/new_data.csv")
    predictions = predict(model, new_data)
    print("Predictions:", predictions)