import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import exp, log, sqrt

import os
os.environ['KAGGLE_USERNAME'] = "chillyon"
os.environ['KAGGLE_KEY'] = "a1af1706a9493950a19751152069f960"
from kaggle.api.kaggle_api_extended import KaggleApi

# Download and Load the dataset from Kaggle
api = KaggleApi()
api.authenticate()
dataset_name = 'miguelcorraljr/ted-ultimate-dataset'
api.dataset_download_files(dataset_name, path='./', unzip=True)

import zipfile
with zipfile.ZipFile('./ted-ultimate-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('./')


# Load Titanic dataset from Kaggle
train_data_en = pd.read_csv("./2020-05-01/ted_talks_en.csv")
#test_data = pd.read_csv("./test.csv")
