# Fake News Classifier

A machine-learning project that classifies a news article as **Fake** or **Real** using its text.

The project uses TF-IDF text features and compares multiple classification algorithms. The current best-performing model is a **Linear Support Vector Machine (Linear SVM)**.

> **Important limitation:** This is an educational text-classification model trained on a particular dataset. It does not verify claims against reliable sources and should not be treated as a fact-checking system. Its predictions may reflect dataset-specific writing, publisher, formatting, or topic patterns.

## Model performance

On the held-out test split (20%, stratified, `random_state=42`), the current Linear SVM achieved:

- **Accuracy:** 98.55%
- **Test examples:** 7,724
- **Incorrect predictions:** 112

Other models evaluated on the same split:

| Model | Accuracy |
|---|---:|
| Linear SVM | **98.55%** |
| Logistic Regression | 97.99% |
| Random Forest | 97.82% |
| XGBoost | 97.26% |
| Multinomial Naive Bayes | 94.12% |

These are results on this dataset's random test split, not a guarantee of performance on new sources or real-world misinformation.

## Dataset

The project uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

The original files contain `title`, `text`, `subject`, and `date`. The model uses article text (title plus body); `subject` and `date` are excluded from the model features.

Labels:
- `0` = Fake
- `1` = Real

The dataset is not included in this repository. Download it from Kaggle and keep the raw files locally if you need to reproduce training.

## How the model works

The deployed artifact should be a single scikit-learn Pipeline containing:

1. `TfidfVectorizer` — transforms article text into TF-IDF features.
2. `LinearSVC` — predicts the class from those features.

Saving the complete pipeline means the application does not need to fit a new vectorizer. It loads the fitted vectorizer and classifier together.

## Repository structure

A suggested structure:

```text
fake-news-classifier/
├── app.py
├── fake_news_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/
    └── fake_news_classification.ipynb
```

The model filename in this README is the expected deployment filename. If your saved model has a different name or location, update `MODEL_PATH` in `app.py` accordingly.

## Run locally

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd fake-news-classifier
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

**Compatibility note:** scikit-learn does not guarantee that a model saved with one version can be loaded with every other version. Use the same scikit-learn version used when saving `fake_news_model.pkl`. For reproducible deployment, pin the exact versions of scikit-learn, NumPy, SciPy, and joblib from the training environment in `requirements.txt`.

### 4. Save the trained pipeline

After training the final pipeline:

```python
import joblib

joblib.dump(svm_pipeline, "fake_news_model.pkl")
```

Keep the model file at the repository root, or update the path in the app.

### 5. Start Streamlit

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Ensure `app.py`, `requirements.txt`, and `fake_news_model.pkl` are committed.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/).
4. Connect your GitHub account and select the repository and branch.
5. Set the app entrypoint to `app.py`.
6. Deploy.

The cloud environment installs packages from `requirements.txt` and runs the app. The model file must be present in the repository (or retrieved through a separately configured storage mechanism).

## Security and responsible use

- Only load trusted model files. Pickle/joblib files can execute code during loading.
- Do not describe a prediction as proof that an article is true or false.
- Avoid presenting an SVM decision score as a calibrated probability or confidence percentage. `LinearSVC` does not provide `predict_proba()` by default.
- Encourage users to verify important claims using multiple reliable sources.

## Future improvements

- Cross-validation and hyperparameter tuning.
- Error analysis on incorrectly classified examples.
- Testing on articles from sources and time periods not represented in the training data.
- A more detailed explanation of model limitations in the interface.
