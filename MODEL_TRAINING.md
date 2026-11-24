# Revio AI Intelligence Engine — Model Training & Evaluation

This document outlines the training workflow, feature pipeline, evaluation summary, and limitations for the Revio AI sentiment classification model[cite: 1].

## Training Methodology

Model training and exploratory analysis live under `src/training/` (`train_model.py` and `eda.py`), with EDA reports saved under `reports/eda/`[cite: 1].

1. **Preprocessing**: Lowercasing, punctuation stripping, and whitespace normalization[cite: 1].
2. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer with unigrams and bigrams (`ngram_range=(1, 2)`), capped at `5,000` max features.
3. **Data Splitting**: Stratified `70%` training, `15%` validation, and `15%` held-out test split to prevent data leakage.
4. **Model Comparison**: Evaluated Logistic Regression, Linear Support Vector Classifier (`LinearSVC`), and Multinomial Naive Bayes.
5. **Selection**: TF-IDF + Logistic Regression was selected based on macro F1 and overall classification performance, exported to `models/best_review_model.joblib` along with `models/model_metadata.json`[cite: 1].

## Evaluation Results (Test Set)

- **Test Accuracy**: `100%` (on the synthetic evaluation benchmark).
- **Macro Precision / Recall / F1**: `1.00`.

## Runtime Inference & NLP Context

- Live batch scoring is served via `ReviewPredictor` (`src/inference/predictor.py`), expecting structured DataFrame inputs with `review_text` during batch prediction[cite: 1].
- Additional NLP utilities under `src/nlp/` (`preprocessor.py`, `aspect_extractor.py`, `emotion_analyzer.py`, `topic_clusterer.py`, `opportunity_engine.py`) and upload normalization/fallback logic support aspect, emotion, priority, and confidence handling[cite: 1].

## Limitations Disclaimer

- Models trained primarily on synthetic or benchmark datasets serve as strong structural demonstrations and may experience domain shift, sarcasm, slang, or confidence variation on real-world customer feedback[cite: 1].
- Production enterprise deployments should evaluate on real domain data and retrain or fine-tune models as needed.