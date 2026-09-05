# Phishing Email Detection Model

A machine learning model built with **Scikit-learn** that classifies emails as
**Phishing** or **Safe**.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Creates the training dataset (`data/emails_dataset.csv`) |
| `phishing_detector.py` | Trains the model and evaluates accuracy |
| `check_my_email.py` | Paste any email text and check if it's phishing or safe |
| `data/emails_dataset.csv` | Training data (360 emails: 180 phishing, 180 safe) Create subfolder |
| `confusion_matrix.png` | Model performance visualization |

## How to Run

```bash
pip install pandas scikit-learn matplotlib scipy

python3 generate_dataset.py     # creates the dataset
python3 phishing_detector.py    # trains and evaluates the model
```

## Checking Your Own Email

Open `check_my_email.py`, paste your email's text into `MY_EMAIL_TEXT`, save, then run:

```bash
python3 check_my_email.py
```

It will print whether the email looks like `phishing` or `safe`, with a confidence percentage.
