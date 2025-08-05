# הרצת מודלים logistic_XGBoost עם ניתוח רגישות
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize

output_dir = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\Model\results\logistic_xgboost"
os.makedirs(output_dir, exist_ok=True)

fakenews_df = pd.read_csv(r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv")
politifact_df = pd.read_csv(r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv")

# פונקציות עזר
def apply_pca(X, n_components=100):
    return PCA(n_components=n_components).fit_transform(X)

def vectorize_text(df, method='tfidf'):
    texts = df['clean_text'].astype(str).tolist()
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=1000)
        return vectorizer.fit_transform(texts).toarray()
    tokenized = [word_tokenize(text) for text in texts]
    if method == 'word2vec':
        model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4, seed=42)
    elif method == 'fasttext':
        model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4, seed=42)
    return np.array([
        np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(100)], axis=0)
        for words in tokenized
    ])

def evaluate_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'xgboost':
        model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'f1_score': round(f1_score(y_test, y_pred), 4)
    }

models = ['logistic', 'xgboost']
methods = ['tfidf', 'word2vec', 'fasttext']
datasets = {
    'FakeNewsNet': fakenews_df,
    'PolitiFact': politifact_df
}

for model_name in models:
    model_results = {}
    for dataset_name, df in datasets.items():
        y = df['label'].values
        emotions_ohe = pd.get_dummies(df['emotion'], prefix='emo')
        for method in methods:
            X_text = vectorize_text(df, method)
            X = np.hstack((X_text, emotions_ohe.values))
            for with_pca in [False, True]:
                X_used = apply_pca(X) if with_pca else X
                config = f"{dataset_name} - {model_name.upper()} - {method.upper()} + Emotions{' + PCA' if with_pca else ''}"
                model_results[config] = evaluate_model(X_used, y, model_name)

    # שמירה לפי מודל
    json_path = os.path.join(output_dir, f"{model_name}_emotion_results.json")
    csv_path = os.path.join(output_dir, f"{model_name}_emotion_results.csv")

    with open(json_path, 'w') as f:
        json.dump(model_results, f, indent=4)

    pd.DataFrame.from_dict(model_results, orient='index').to_csv(csv_path)

    print(f"\n📁 Results for {model_name.upper()} saved to:\n{json_path}\n{csv_path}")
