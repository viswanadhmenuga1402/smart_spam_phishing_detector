# Smart Spam & Phishing Detector

A beginner-friendly Python and Machine Learning project for an internship portfolio. It classifies messages as **Spam** or **Ham** using TF-IDF and Multinomial Naive Bayes. A separate rule-based checker flags a few suspicious patterns for review.

## Features

- Streamlit web application
- Text classification with scikit-learn
- TF-IDF feature extraction
- Naive Bayes model
- Accuracy, precision, recall, F1-score and confusion matrix
- Rule-based indicators for suspicious message patterns
- CSV scan history and download
- Input validation and error handling

## Tech stack

Python, pandas, scikit-learn, Streamlit, CSV, regular expressions.

## Run on Windows

Open Command Prompt in this project folder and run:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py` is unavailable but `python` works, create the environment with `python -m venv .venv`.

## Dataset

`data/messages.csv` is a small illustrative dataset included so the project can run immediately. It is not large enough to support real-world accuracy claims. For a serious evaluation, use a larger, legally obtained, representative and carefully labeled dataset, then test on examples not used during training.

## How it works

1. `load_dataset()` validates the CSV columns and labels.
2. `train_model()` splits the data and trains a TF-IDF + Multinomial Naive Bayes model.
3. `predict_message()` predicts the label and returns the model's predicted-class probability.
4. `inspect_risk()` independently checks for configured warning patterns.
5. `save_scan()` records the result to `data/scan_history.csv`.

## Python concepts demonstrated

- Variables, strings, lists, dictionaries and conditions
- Functions and modules
- Imports and package management
- File handling with CSV
- Exceptions and input validation
- Regular expressions
- Data processing with pandas
- Reusable functions and separation of concerns

## Machine Learning concepts demonstrated

- Supervised text classification
- Labeled data (`Ham` / `Spam`)
- Train/test split and stratification
- TF-IDF vectorization and word n-grams
- Multinomial Naive Bayes
- Precision, recall, F1-score, accuracy and confusion matrix

## Limitations and responsible use

This is an educational demonstration, not a production email security product. The sample data is small and artificial; model probabilities are not guarantees. The rule-based risk checker is not a trained phishing detector and can miss attacks or flag legitimate messages. It does not open links, inspect attachments, verify senders, or replace security tools. Avoid pasting private or confidential messages into the demo.

## Suggested resume description

**Smart Spam & Phishing Detector | Python, scikit-learn, pandas, Streamlit**
- Built a web-based text classifier using TF-IDF features and Multinomial Naive Bayes to categorize messages as spam or legitimate.
- Evaluated the model using precision, recall, F1-score, accuracy, and a confusion matrix.
- Implemented an independent rule-based warning checker and CSV-based scan history.

Only include metrics you actually measured on an appropriate test dataset. Be prepared to explain the dataset, train/test split, false positives, false negatives, and project limitations.
