import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from openTSNE import TSNE
import numpy as np

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

def classDistVisualize(method, train_data, eval_data, perplexity = 30, figsize=(12, 6)):
    train_features, train_labels = train_data[0], train_data[1]
    eval_features, eval_labels = eval_data[0], eval_data[1]

    # Convert labels: 0 → "Negative", 1 → "Positive"
    label_map = {0: "Negative", 1: "Positive"}
    train_labels = np.array([label_map[label] for label in train_labels])
    eval_labels = np.array([label_map[label] for label in eval_labels])

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Standardize the train and test data (both should be standardized using the same scaler)
    scaled_train_features = scaler.fit_transform(train_features)  # Fit and transform train data
    scaled_eval_features = scaler.transform(eval_features)  # Only transform the test data
    # Apply PCA/TSNE to both train and test data
    projector = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, perplexity = perplexity)
    if (method == 'pca'):
        projector = PCA(n_components=2)
        projected_train_features = projector.fit_transform(scaled_train_features)
        projected_eval_features = projector.transform(scaled_eval_features)
    else:
        projector = TSNE(perplexity=perplexity, metric="euclidean", n_jobs=8, random_state=42, verbose=True)
        projected_train_features = projector.fit(scaled_train_features)
        projected_eval_features = projected_train_features.transform(scaled_eval_features)

    # Create a plot
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Set axis labels
    axis_label = ['Principal Component 1', 'Principal Component 2'] if method == 'pca' else ['t-SNE 1', 't-SNE 2']

    # Set method
    method = 'PCA' if method == 'pca' else 't-SNE'

    # Scatter plot
    sns.scatterplot(x=projected_train_features[:, 0], y=projected_train_features[:, 1], hue=train_labels, alpha=0.7, palette='viridis', ax=axes[0], hue_order=['Positive', 'Negative'])
    axes[0].set_title(f'Distribution of Training samples using {method}')
    axes[0].set_xlabel(axis_label[0])
    axes[0].set_ylabel(axis_label[1])

    # Add "A" inside the first plot
    xmin, xmax = axes[0].get_xlim()
    ymin, ymax = axes[0].get_ylim()
    axes[0].text(xmin + 0.025 * (xmax - xmin), ymax - 0.025 * (ymax - ymin), 'A',
                 fontsize=20, fontweight='bold', color='black', ha='left', va='top')

    sns.scatterplot(x=projected_eval_features[:, 0], y=projected_eval_features[:, 1], hue=eval_labels, alpha=0.7, palette='viridis', ax=axes[1], hue_order=['Positive', 'Negative'])
    axes[1].set_title(f'Distribution of Test samples using {method}')
    axes[1].set_xlabel(axis_label[0])
    axes[1].set_ylabel(axis_label[1])

    # Add "B" inside the second plot
    xmin, xmax = axes[1].get_xlim()
    ymin, ymax = axes[1].get_ylim()
    axes[1].text(xmin + 0.025 * (xmax - xmin), ymax - 0.025 * (ymax - ymin), 'B',
                 fontsize=20, fontweight='bold', color='black', ha='left', va='top')

    # Set axis labels
    axis_label = ['Principal Component 1', 'Principal Component 2'] if method == 'PCA' else ['t-SNE 1', 't-SNE 2']

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("data/processed/processed_data.csv")
    plot_correlation_matrix(data, save_path="reports/figures/correlation_matrix.png")
    plot_feature_distribution(data, feature="feature1", save_path="reports/figures/feature1_distribution.png")