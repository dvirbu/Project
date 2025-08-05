# מודל SVM

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize
import nltk


FAKENEWS_PATH = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv"
POLITIFACT_PATH = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv"


datasets = {
    "FakeNewsNet": pd.read_csv(FAKENEWS_PATH),
    "PolitiFact": pd.read_csv(POLITIFACT_PATH)
}

results = {}


def run_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "accuracy": report['accuracy'],
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1_score": report['weighted avg']['f1-score']
    }


def get_sentence_vectors(texts, model, dim):
    vectors = []
    for text in texts:
        tokens = word_tokenize(text)
        word_vecs = [model.wv[w] for w in tokens if w in model.wv]
        if word_vecs:
            vec = np.mean(word_vecs, axis=0)
        else:
            vec = np.zeros(dim)
        vectors.append(vec)
    return np.array(vectors)


for name, df in datasets.items():
    clean_texts = df['clean_text'].astype(str).tolist()
    labels = df['label'].values

    results[name] = {}

    # 1. TF-IDF בלבד
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(clean_texts).toarray()
    results[name]['TF-IDF'] = run_svm(X_tfidf, labels)

    # 2. TF-IDF + PCA
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X_tfidf)
    results[name]['TF-IDF + PCA'] = run_svm(X_pca, labels)

    # 3. Word2Vec בלבד
    tokenized = [word_tokenize(t) for t in clean_texts]
    w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    X_w2v = get_sentence_vectors(clean_texts, w2v_model, 100)
    results[name]['Word2Vec'] = run_svm(X_w2v, labels)

    # 4. Word2Vec + PCA
    X_w2v_pca = PCA(n_components=50).fit_transform(X_w2v)
    results[name]['Word2Vec + PCA'] = run_svm(X_w2v_pca, labels)

    # 5. FastText בלבד
    ft_model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    X_ft = get_sentence_vectors(clean_texts, ft_model, 100)
    results[name]['FastText'] = run_svm(X_ft, labels)

    # 6. FastText + PCA
    X_ft_pca = PCA(n_components=50).fit_transform(X_ft)
    results[name]['FastText + PCA'] = run_svm(X_ft_pca, labels)

# שמירת תוצאות
with open("svm_comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n Results saved to 'svm_comparison_results.json' ")

