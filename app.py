"""Streamlit application for the Fake News Classifier.

Loads a pre-trained scikit-learn Pipeline (TF-IDF Vectorizer + LinearSVC)
and provides an accessible, restrained interface to classify news articles
as Fake News or Real News.
"""

from pathlib import Path
import logging
from typing import Optional, Any

import streamlit as st
import joblib

# Set up logging for developer diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fake_news_app")

# Class mapping according to project specifications:
# 0 = Fake, 1 = Real
CLASS_MAPPING = {
    0: "Fake News",
    1: "Real News",
}

DISCLAIMER_TEXT = (
    "This is a machine-learning prediction based on patterns learned from the "
    "training dataset. It is not independent fact verification. Please verify "
    "important claims using reliable sources."
)

GITHUB_URL = "https://github.com/Sqrvesh/news-credibility-classifier"


def get_model_path() -> Optional[Path]:
    """Resolve the model file path relative to app.py.

    Checks expected deployment filenames and locations in order.
    """
    base_dir = Path(__file__).resolve().parent
    candidate_paths = [
        base_dir / "fake_news_model.pkl",
        base_dir / "model" / "news_classifier_model.pkl",
        base_dir / "model" / "fake_news_model.pkl",
        base_dir / "news_classifier_model.pkl",
    ]

    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate

    return None


@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> Any:
    """Load and cache the serialized scikit-learn pipeline using joblib."""
    try:
        logger.info("Loading model pipeline from: %s", path)
        model = joblib.load(path)
        logger.info("Successfully loaded model pipeline: %s", type(model))
        return model
    except Exception as exc:
        logger.exception("Failed to load model from %s: %s", path, exc)
        raise RuntimeError(
            f"Could not load model pipeline from '{path.name}'. "
            f"Details: {exc}"
        ) from exc


def main() -> None:
    # Page setup
    st.set_page_config(
        page_title="Fake News Classifier",
        page_icon="📰",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Restrained styling: neutral palette, clean typography, dimmed cards & icon button
    st.markdown(
        """
        <style>
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.35rem;
        }
        .main-header {
            margin: 0;
            padding: 0;
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .github-icon-btn,
        .github-icon-btn:link,
        .github-icon-btn:visited {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 38px;
            height: 38px;
            border-radius: 8px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background-color: transparent;
            color: #8b949e !important;
            text-decoration: none;
            transition: background-color 0.2s ease, border-color 0.2s ease;
            cursor: pointer;
            flex-shrink: 0;
        }
        .github-icon-btn:hover {
            background-color: rgba(128, 128, 128, 0.15);
            border-color: rgba(128, 128, 128, 0.45);
            color: #ffffff !important;
        }
        .github-icon-btn svg {
            width: 20px;
            height: 20px;
            fill: #8b949e !important;
            transition: fill 0.2s ease;
        }
        .github-icon-btn:hover svg {
            fill: #ffffff !important;
        }
        .sub-header {
            opacity: 0.75;
            font-size: 1.05rem;
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }

        /* Solid toned-down red Classify Button that lightens on hover */
        div.stButton > button,
        button[data-testid="stBaseButton-primary"],
        button[data-testid="stBaseButton-secondary"] {
            background-color: #c03939 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            padding: 0.6rem 1.25rem !important;
            transition: background-color 0.2s ease !important;
            box-shadow: none !important;
        }
        div.stButton > button:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #bd4848!important;
            color: #ffffff !important;
        }
        div.stButton > button:active,
        button[data-testid="stBaseButton-primary"]:active,
        button[data-testid="stBaseButton-secondary"]:active {
            background-color: #942424 !important;
        }
        div.stButton > button:focus,
        div.stButton > button:focus-visible {
            box-shadow: 0 0 0 2px rgba(168, 45, 45, 0.4) !important;
            outline: none !important;
        }

        .result-box {
            padding: 1.25rem 1.5rem;
            border-radius: 8px;
            margin-top: 1.25rem;
            margin-bottom: 1rem;
            border-left: 5px solid;
        }
        .result-real {
            background-color: rgba(46, 125, 50, 0.12);
            border-color: #2e7d32;
        }
        .result-real .result-label {
            color: #2e7d32;
        }
        .result-fake {
            background-color: rgba(198, 40, 40, 0.12);
            border-color: #c62828;
        }
        .result-fake .result-label {
            color: #c62828;
        }
        @media (prefers-color-scheme: dark) {
            .result-real {
                background-color: rgba(46, 125, 50, 0.18);
                border-color: #4caf50;
            }
            .result-real .result-label {
                color: #66bb6a;
            }
            .result-fake {
                background-color: rgba(198, 40, 40, 0.18);
                border-color: #ef5350;
            }
            .result-fake .result-label {
                color: #ef5350;
            }
        }
        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
            opacity: 0.7;
        }
        .result-label {
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .disclaimer-box {
            font-size: 0.82rem;
            color: #7b838f;
            background-color: transparent;
            margin-top: 1.75rem;
            line-height: 1.5;
        }
        .app-footer {
            margin-top: 1rem;
            padding-top: 0.6rem;
            border-top: 1px solid rgba(128, 128, 128, 0.15);
            font-size: 0.85rem;
            color: #7b838f;
            line-height: 1.6;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header section with clean, responsive GitHub icon button
    st.markdown(
        f"""
        <div class="header-container">
            <h1 class="main-header">Fake News Classifier</h1>
            <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer" class="github-icon-btn" title="View repository on GitHub" aria-label="GitHub Repository">
                <svg viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
                </svg>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="sub-header">'
        "Submit article text below to obtain a machine-learning classification based on "
        "text patterns learned from the training dataset."
        "</p>",
        unsafe_allow_html=True,
    )

    # Locate and load the pre-trained model
    model_path = get_model_path()
    if model_path is None:
        st.error(
            "**Model file not found.**\n\n"
            "Please ensure `fake_news_model.pkl` is present in the application root directory "
            "or `model/news_classifier_model.pkl` exists."
        )
        logger.error("No model file found in candidate locations relative to app.py.")
        return

    try:
        model = load_model(model_path)
    except Exception as err:
        st.error(
            f"**Error loading model:** An issue occurred while reading the model file.\n\n"
            f"Please verify that the model file is valid and compatible with the current environment.\n\n"
            f"`{err}`"
        )
        return

    # Text input area
    article_text = st.text_area(
        label="Article Text",
        height=220,
        placeholder="Paste headline and article text here...",
        help="Input the raw article title and body for classification.",
    )

    # Action button
    classify_clicked = st.button("Classify Article", use_container_width=True)

    if classify_clicked:
        clean_text = article_text.strip()
        if not clean_text:
            st.warning("Please enter some article text to classify.")
        else:
            try:
                # Predict directly using the full scikit-learn pipeline
                # The pipeline handles raw text via its fitted TfidfVectorizer
                pred_label_id = model.predict([article_text])[0]
                label_name = CLASS_MAPPING.get(int(pred_label_id), f"Class {pred_label_id}")
                css_class = "result-real" if int(pred_label_id) == 1 else "result-fake"

                # Display result with dimmed background and no leading icons
                st.markdown(
                    f"""
                    <div class="result-box {css_class}">
                        <div class="result-title">Classification Result</div>
                        <div class="result-label">{label_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as exc:
                logger.exception("Prediction failed: %s", exc)
                st.error(
                    "An error occurred while generating the prediction. "
                    "Please check the application logs for details."
                )

    # Greyed-out notice section without bold prefix
    st.markdown(
        f"""
        <div class="disclaimer-box">
            {DISCLAIMER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Left-aligned footer without GitHub link
    st.markdown(
        """
        <div class="app-footer">
            Sarvesh M &bull; 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
