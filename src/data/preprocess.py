def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()
    return text

df['cleaned_transcript'] = df['transcript'].apply(clean_text)

df = df[['cleaned_transcript']]        # Only keep cleaned transcript
print(f"Language en:")
print(df.head())