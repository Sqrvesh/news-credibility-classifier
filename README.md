# News Credibility Classifier

A machine-learning project that classifies a news article as **Fake** or **Real** using its text.

The project uses TF-IDF text features and compares multiple classification algorithms. The current best-performing model is a Linear Support Vector Machine (Linear SVM).

Live at https://news-credibility-classifier-sarvesh.streamlit.app/

> This is a text-classification model trained on a particular dataset. It does not verify claims against reliable sources and should not be treated as a fact-checking system. Its predictions may reflect dataset-specific writing, publisher, formatting, or topic patterns.

## Model performance

The Linear SVM achieved an accuracy of 98.55%

Other models evaluated on the same train test split:

| Model | Accuracy |
|---|---:|
| Linear SVM | **98.55%** |
| Logistic Regression | 97.99% |
| Random Forest | 97.82% |
| XGBoost | 97.26% |
| Multinomial Naive Bayes | 94.12% |


## Dataset

The project uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

The original files contain `title`, `text`, `subject`, and `date`. The model uses article text (title plus body); `subject` and `date` are excluded from the model features.

Labels:
- `0` = Fake
- `1` = Real


## How the model works

We run a scikit-learn Pipeline containing:

1. `TfidfVectorizer` - transforms article text into TF-IDF features. (Preprocessing)
2. `LinearSVC` - predicts the class from those features. (Training)



## Run locally

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd news-credibility-classifier
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Streamlit

```bash
streamlit run app.py
```
