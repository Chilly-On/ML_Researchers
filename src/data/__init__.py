train_data = pd.read_csv(f"./2020-05-01/ted_talks_en.csv")
df = pd.DataFrame(train_data)
print("\nData converted to DataFrame:")
# Select the transcription column
df = df[['transcript']]
# Clean the null in the data
df = df.dropna(subset=['transcript'])
print(df.head())