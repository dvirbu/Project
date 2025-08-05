
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec, FastText
from sklearn.preprocessing import StandardScaler

def vectorize_word2vec(texts, model, dim):
    return np.array([
        np.mean([model.wv[w] for w in tokens if w in model.wv] or [np.zeros(dim)], axis=0)
        for tokens in texts
    ])

def process_emotion_features(df_emotions_column, all_possible_emotions):
    if df_emotions_column.empty:
        return pd.DataFrame(index=df_emotions_column.index)
    df_emotions_column_categorical = pd.Categorical(df_emotions_column, categories=all_possible_emotions)
    return pd.get_dummies(df_emotions_column_categorical, prefix='emotion')

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, preds, average="weighted", zero_division=0)
    }

fakenews_df = pd.read_csv("preprocessed_fakenewsnet_balanced.csv")
politifact_df = pd.read_csv("preprocessed_politifact_balanced.csv")

datasets = {"FakeNewsNet": fakenews_df, "PolitiFact": politifact_df}
ALL_POSSIBLE_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
final_results = {}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for dataset_name, df in datasets.items():
    if 'emotion' not in df.columns:
        df['emotion'] = 'neutral'
    df = df.dropna(subset=["clean_text", "label", "emotion"])

    texts = df["clean_text"].astype(str).values
    labels = df["label"].values
    emotions_original_series = df["emotion"]
    emotions_ohe_full_df = process_emotion_features(emotions_original_series, ALL_POSSIBLE_EMOTIONS)

    config_scores = {
        "TF-IDF + Emotions": [],
        "TF-IDF + Emotions + PCA": [],
        "Word2Vec + Emotions": [],
        "Word2Vec + Emotions + PCA": [],
        "FastText + Emotions": [],
        "FastText + Emotions + PCA": []
    }

    for fold, (train_index, test_index) in enumerate(skf.split(texts, labels)):
        X_train_texts, X_test_texts = texts[train_index], texts[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        X_train_emotions = emotions_ohe_full_df.iloc[train_index].values
        X_test_emotions = emotions_ohe_full_df.iloc[test_index].values

        base_models = [("svm", SVC(probability=True, random_state=42)),
                       ("rf", RandomForestClassifier(random_state=42)),
                       ("lr", LogisticRegression(max_iter=1000, random_state=42))]
        meta_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
        stack_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)

        # TF-IDF + Emotions
        tfidf = TfidfVectorizer(max_features=1000)
        X_train_tfidf = tfidf.fit_transform(X_train_texts).toarray()
        X_test_tfidf = tfidf.transform(X_test_texts).toarray()
        config_scores["TF-IDF + Emotions"].append(
            evaluate_model(stack_model, np.hstack((X_train_tfidf, X_train_emotions)),
                                         np.hstack((X_test_tfidf, X_test_emotions)), y_train, y_test))

        # TF-IDF + Emotions + PCA
        pca = PCA(n_components=100, random_state=42)
        X_train_pca = pca.fit_transform(np.hstack((X_train_tfidf, X_train_emotions)))
        X_test_pca = pca.transform(np.hstack((X_test_tfidf, X_test_emotions)))
        config_scores["TF-IDF + Emotions + PCA"].append(
            evaluate_model(stack_model, X_train_pca, X_test_pca, y_train, y_test))

        # Word2Vec + Emotions
        w2v = Word2Vec(sentences=[t.split() for t in X_train_texts], vector_size=100, window=5, min_count=1, workers=4)
        X_train_w2v = vectorize_word2vec([t.split() for t in X_train_texts], w2v, 100)
        X_test_w2v = vectorize_word2vec([t.split() for t in X_test_texts], w2v, 100)
        scaler = StandardScaler()
        X_train_w2v_scaled = scaler.fit_transform(X_train_w2v)
        X_test_w2v_scaled = scaler.transform(X_test_w2v)
        config_scores["Word2Vec + Emotions"].append(
            evaluate_model(stack_model, np.hstack((X_train_w2v_scaled, X_train_emotions)),
                                         np.hstack((X_test_w2v_scaled, X_test_emotions)), y_train, y_test))

        # Word2Vec + Emotions + PCA
        pca = PCA(n_components=100, random_state=42)
        X_train_pca = pca.fit_transform(np.hstack((X_train_w2v_scaled, X_train_emotions)))
        X_test_pca = pca.transform(np.hstack((X_test_w2v_scaled, X_test_emotions)))
        config_scores["Word2Vec + Emotions + PCA"].append(
            evaluate_model(stack_model, X_train_pca, X_test_pca, y_train, y_test))

        # FastText + Emotions
        ft = FastText(sentences=[t.split() for t in X_train_texts], vector_size=100, window=5, min_count=1, workers=4)
        X_train_ft = vectorize_word2vec([t.split() for t in X_train_texts], ft, 100)
        X_test_ft = vectorize_word2vec([t.split() for t in X_test_texts], ft, 100)
        scaler = StandardScaler()
        X_train_ft_scaled = scaler.fit_transform(X_train_ft)
        X_test_ft_scaled = scaler.transform(X_test_ft)
        config_scores["FastText + Emotions"].append(
            evaluate_model(stack_model, np.hstack((X_train_ft_scaled, X_train_emotions)),
                                         np.hstack((X_test_ft_scaled, X_test_emotions)), y_train, y_test))

        # FastText + Emotions + PCA
        pca = PCA(n_components=100, random_state=42)
        X_train_pca = pca.fit_transform(np.hstack((X_train_ft_scaled, X_train_emotions)))
        X_test_pca = pca.transform(np.hstack((X_test_ft_scaled, X_test_emotions)))
        config_scores["FastText + Emotions + PCA"].append(
            evaluate_model(stack_model, X_train_pca, X_test_pca, y_train, y_test))

    for config, scores in config_scores.items():
        metrics = {k: [s[k] for s in scores] for k in scores[0]}
        result = {}
        for metric, values in metrics.items():
            result[f"{metric}_mean"] = np.mean(values)
            result[f"{metric}_std"] = np.std(values)
        final_results[f"{dataset_name} - {config}"] = result

with open("emotions_only_results.json", "w") as f:
    json.dump(final_results, f, indent=4)

from google.colab import files
files.download("emotions_only_results.json")
