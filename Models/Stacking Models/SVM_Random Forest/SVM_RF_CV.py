import pandas as pd
import numpy as np
import json
from sklearn.model_selection import cross_validate
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import nltk

def get_word2vec_embeddings(texts, method='word2vec'):
    tokenized = [word_tokenize(text) for text in texts]
    if method == 'word2vec':
        model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    elif method == 'fasttext':
        model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    else:
        raise ValueError("Method must be 'word2vec' or 'fasttext'.")

    vectors = []
    for tokens in tokenized:
        word_vecs = [model.wv[token] for token in tokens if token in model.wv]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)


def evaluate_model(X, y, dataset_name, method_name, results_dict):
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ]
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3
    )

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }

    scores = cross_validate(clf, X, y, cv=3, scoring=scoring)

    results_dict[f"{dataset_name} - {method_name}"] = {
        'accuracy': np.mean(scores['test_accuracy']),
        'precision': np.mean(scores['test_precision']),
        'recall': np.mean(scores['test_recall']),
        'f1_score': np.mean(scores['test_f1'])
    }


results = {}

DATASETS = {
    'FakeNewsNet': r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv",
    'PolitiFact': r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv"
}

for dataset_name, path in DATASETS.items():
    df = pd.read_csv(path).dropna()
    texts = df['clean_text'].astype(str).tolist()
    labels = df['label'].values

    # 1. TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(texts).toarray()
    evaluate_model(X_tfidf, labels, dataset_name, "TF-IDF", results)

    # 2. TF-IDF + PCA
    X_tfidf_pca = PCA(n_components=100).fit_transform(X_tfidf)
    evaluate_model(X_tfidf_pca, labels, dataset_name, "TF-IDF + PCA", results)

    # 3. Word2Vec
    X_w2v = get_word2vec_embeddings(texts, method='word2vec')
    evaluate_model(X_w2v, labels, dataset_name, "Word2Vec", results)

    # 4. Word2Vec + PCA
    X_w2v_pca = PCA(n_components=50).fit_transform(X_w2v)
    evaluate_model(X_w2v_pca, labels, dataset_name, "Word2Vec + PCA", results)

    # 5. FastText
    X_ft = get_word2vec_embeddings(texts, method='fasttext')
    evaluate_model(X_ft, labels, dataset_name, "FastText", results)

    # 6. FastText + PCA
    X_ft_pca = PCA(n_components=50).fit_transform(X_ft)
    evaluate_model(X_ft_pca, labels, dataset_name, "FastText + PCA", results)

with open("stacking_cv3_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Results saved to 'stacking_cv3_results.json'")
