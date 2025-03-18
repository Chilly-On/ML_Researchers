import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def create_features(df, text_column='cleaned_transcript', max_features=5000, ngram_range=(1, 2)):
    """
    Vectorize text data using TF-IDF.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text data.
        text_column (str): Name of the column containing the text data.
        max_features (int): Maximum number of features (words) to consider.
        ngram_range (tuple): Range of n-grams to use (e.g., (1, 2) for unigrams and bigrams).

    Returns:
        pd.DataFrame: DataFrame with TF-IDF features and the target column.
    """
    # Split data into features and target variable
    X = df[text_column]

    # Vectorizing text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_vectorized = vectorizer.fit_transform(X)

    # Convert to dense array
    X_dense = X_vectorized.toarray()

    # Create DataFrame with words as columns
    df_tfidf = pd.DataFrame(X_dense, columns=vectorizer.get_feature_names_out())

    # Add target column if it exists in the original DataFrame
    if 'target' in df.columns:
        df_tfidf['target'] = df['target'].values
    else:
        df_tfidf['target'] = None  # Placeholder if target column is not present

    print(f"Shape of vectorized data: {X_vectorized.shape}")
    print(f"Shape of TF-IDF DataFrame: {df_tfidf.shape}")
    print("Sample of TF-IDF DataFrame:")
    print(df_tfidf.head())

    return df_tfidf