import pandas as pd
import os

def clean_data(data, text_column='transcript'):
    """
    Clean the data by removing null values and duplicates.
    
    Args:
        data (pd.DataFrame): Data to clean.
        text_column (str): Name of the column containing text data.
    
    Returns:
        pd.DataFrame: Cleaned data.
    """
    # Check if the text column exists
    if text_column not in data.columns:
        raise ValueError(f"Column '{text_column}' not found in the data.")
    
    # Remove rows with null values in the text column
    data = data.dropna(subset=[text_column])
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    return data

def preprocess_data(data, text_column='transcript'):
    """
    Preprocess the data by cleaning and transforming it.
    
    Args:
        data (pd.DataFrame): Data to preprocess.
        text_column (str): Name of the column containing text data.
    
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Clean the data
    data = clean_data(data, text_column=text_column)
    
    # Add additional preprocessing steps here if needed
    # Example: Convert text to lowercase
    data[text_column] = data[text_column].str.lower()
    
    return data

def save_processed_data(data, filepath):
    """
    Save processed data to a file.
    
    Args:
        data (pd.DataFrame): Processed data to save.
        filepath (str): Path to save the processed data.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the data to a CSV file
    data.to_csv(filepath, index=False)







# def clean_text(text):
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\d+', '', text)  # Remove numbers
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     text = text.strip()
#     return text

# df['cleaned_transcript'] = df['transcript'].apply(clean_text)

# df = df[['cleaned_transcript']]        # Only keep cleaned transcript
# print(f"Language en:")
# print(df.head())