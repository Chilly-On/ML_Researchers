import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
# Split data into features and target variable
X = df['cleaned_transcript']

# Vectorizing text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Consider unigrams and bigrams
X_vectorized = vectorizer.fit_transform(X)

# Convert to dense array
X_dense = X_vectorized.toarray()

# Create DataFrame with words as columns
df_tfidf = pd.DataFrame(X_dense, columns=vectorizer.get_feature_names_out())
print(X_vectorized.shape)
print(df_tfidf.shape)
df_tfidf['target'] = None

# Display a sample
print(df_tfidf.head())
