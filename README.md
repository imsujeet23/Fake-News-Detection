# Fake News Detection Using NLP

This Phase 2 project implements a fake news detection pipeline using Natural Language Processing and traditional machine learning. It follows the methodology described in the report: data preprocessing, TF-IDF feature extraction, SMOTE-based balancing, model training with Logistic Regression and Naive Bayes, and performance evaluation using accuracy, precision, recall, and F1-score.

Note: the repository includes a very small starter dataset so the pipeline can run immediately. Replace `data/raw/fake_news.csv` with your full Kaggle dataset for meaningful academic results.

## Project Structure

```text
fake-news-detection/
├── data/
│   ├── raw/fake_news.csv
│   └── processed/cleaned_data.csv
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── predict.py
├── notebooks/fake_news_analysis.ipynb
├── models/trained_model.pkl
├── outputs/
│   ├── confusion_matrix.png
│   └── results.txt
├── requirements.txt
├── README.md
├── main.py
└── .gitignore
```

## Features

- Cleans and normalizes news text
- Builds TF-IDF text features
- Balances training data with SMOTE
- Trains Logistic Regression and Naive Bayes classifiers
- Selects the best model using F1-score
- Saves evaluation results and confusion matrix
- Stores the trained model for later predictions

## Dataset Format

The raw dataset must contain these columns:

- `title`
- `text`
- `label`

Labels should be either `REAL` or `FAKE`.

## Installation

```bash
cd fake-news-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The Project

```bash
python3 main.py
```

## Run The Streamlit App

```bash
streamlit run app.py
```

After training, the project generates:

- `data/processed/cleaned_data.csv`
- `models/trained_model.pkl`
- `outputs/results.txt`
- `outputs/confusion_matrix.png`

## Next Improvements

- Add word embeddings or BERT-based comparison
- Build a Streamlit front end for live predictions
- Add cross-validation and hyperparameter tuning
- Integrate SHAP or LIME for explainability
