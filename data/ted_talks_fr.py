import pandas as pd
import sys

# Sửa lỗi in tiếng Việt trong terminal (Windows)
sys.stdout.reconfigure(encoding="utf-8")

# Đường dẫn file CSV
file_path = r"d:\btl ai\ML_Researchers-main\data\raw\ted_talks_fr.csv"

# Đọc file CSV với UTF-8 và giữ lại các cột cần thiết
df = pd.read_csv(file_path, encoding="utf-8")[["transcript"]].dropna()


df.to_csv("transcripts_fr.csv", index=False)
