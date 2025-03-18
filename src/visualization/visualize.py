import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(data, save_path=None):
    """Plot a correlation matrix for the dataset."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_distribution(data, feature, save_path=None):
    """Plot the distribution of a feature."""
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("data/processed/processed_data.csv")
    plot_correlation_matrix(data, save_path="reports/figures/correlation_matrix.png")
    plot_feature_distribution(data, feature="feature1", save_path="reports/figures/feature1_distribution.png")