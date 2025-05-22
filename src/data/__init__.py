from .make_dataset import load_raw_data, save_raw_data
from .preprocess import clean_data, preprocess_data, save_processed_data
# import pandas as pd


# train_data = pd.read_csv(f"C:\Users\DELL\Desktop\Machine learning\ML_Researchers-data\data\raw\ted_talks_en.csv")
# df = pd.DataFrame(train_data)
# print("\nData converted to DataFrame:")
# # Select the transcription column
# df = df[['transcript']]
# # Clean the null in the data
# df = df.dropna(subset=['transcript'])
# print(df.head())