# ML_Researchers# TED Talk Popularity Prediction

## GitHub Repository Link
[GitHub Repo](https://github.com/Chilly-On/ML_Researchers)

---

## Team Members and Task Distribution
|      Team Member     | Student's ID |                Tasks                    |                               Contributions                                |
|----------------------|--------------|-----------------------------------------|----------------------------------------------------------------------------|
| Đặng Ngọc Phú        |   2252617    |  Model Training, Hyperparameter Tuning  |   Trained and tuned machine learning models using GridSearchCV             |
|                      |              |  Visualization, Report Writing          |   Evaluated model performance                                              |
|                      |              |                                         |   Selected the best model                                                  |
|                      |              |                                         |   Created visualizations                                                   |
|----------------------|--------------|-----------------------------------------|----------------------------------------------------------------------------|
| Phùng Gia Minh Khôi  |   2252381    | Data Preprocessing, Feature Engineering |   Cleaned and preprocessed the dataset                                     |
|                      |              |                                         |   Performed TF-IDF vectorization                                           |
|                      |              |                                         |   Handled missing values                                                   |
|                      |              |                                         |   Removed duplicates                                                       |
|----------------------|--------------|-----------------------------------------|----------------------------------------------------------------------------|
| Nguyễn Lê Khải Trọng |   2252850    | Data Preprocessing, Report Writing      |   Cleaned and preprocessed the dataset                                     |
|                      |              |                                         |   Wrote the project report                                                 |
|                      |              |                                         |   Documented the workflow                                                  |
|----------------------|--------------|-----------------------------------------|----------------------------------------------------------------------------|
| Đoàn Tấn Sang        |   2252711    | Visualization, Report Writing           |   Created visualizations (e.g., correlation matrix, feature distributions) |
|                      |              |                                         |   Wrote the project report                                                 |
|                      |              |                                         |   Documented the workflow                                                  |
|----------------------|--------------|-----------------------------------------|----------------------------------------------------------------------------|
---

## Project Overview
This project aims to predict the popularity of TED Talks based on their transcripts and metadata. The target variable is binary:
- `1`: High-viewed talks (views above the median).
- `0`: Low-viewed talks (views below the median).

We used machine learning models to classify talks into these categories and evaluated their performance using accuracy, precision, recall, and F1-score.

---

## Dataset Description
The dataset contains the following columns:
- `talk_id`: Unique identifier for each talk.
- `title`: Title of the TED Talk.
- `transcript`: Full transcript of the talk.
- `views`: Number of views (used to define the target variable).
- `target`: Binary target variable (`1` for high-viewed, `0` for low-viewed).

### Dataset Source
The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/ted-talks).

### Dataset Preprocessing
- Removed rows with missing values in the `transcript` column.
- Removed duplicate rows.
- Defined the target variable (`target`) based on the median number of views.

---

## Methodology

### 1. Data Preprocessing
- **Handling Missing Values**: Removed rows with missing `transcript` values.
- **Removing Duplicates**: Dropped duplicate rows to ensure data quality.
- **Target Variable Definition**: Created a binary target variable (`target`) based on the median number of views.

### 2. Feature Engineering
- **TF-IDF Vectorization**: Transformed the `transcript` column into numerical features using TF-IDF vectorization with `max_features=5000` and `ngram_range=(1, 2)`.
- **Feature Selection**: Used only the TF-IDF features for model training.

### 3. Model Training
- **Models Used**:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost
- **Hyperparameter Tuning**: Used `GridSearchCV` to tune hyperparameters for each model.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score.

### 4. Model Evaluation
- **Validation Set**: Evaluated models on the validation set to select the best model.
- **Test Set**: Evaluated the best model on the test set for final performance metrics.

---

## Results

### Best Model: Support Vector Machine (SVM)
The best-performing model was **SVM** with the following hyperparameters:
- `C`: 10.0
- `kernel`: `rbf`

### Performance Metrics on Test Set
| Metric     | Score  |
|------------|--------|
| Accuracy   | 0.6704 |
| Precision  | 0.6707 |
| Recall     | 0.6704 |
| F1-Score   | 0.6704 |

---

## How to Run the Code

### 1. Clone the Repository
```bash
git clone https://github.com/Chilly-On/ML_Researchers.git