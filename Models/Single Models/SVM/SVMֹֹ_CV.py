# מודל SVM עם Cross Validation
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize
import nltk


# נתיבים לקבצים
FAKENEWS_PATH = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv"
POLITIFACT_PATH = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv"

datasets = {
    "FakeNewsNet": pd.read_csv(FAKENEWS_PATH),
    "PolitiFact": pd.read_csv(POLITIFACT_PATH)
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

results = {}

for name, df in datasets.items():
    print(f"\n Processing dataset: {name}")
    clean_texts = df['clean_text'].astype(str).tolist()
    labels = df['label'].values
    results[name] = {}

    def evaluate_model(X, model_name):
        model = SVC(kernel='linear', random_state=42)
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        scores = cross_validate(model, X, labels, cv=5, scoring=scoring)

        return {
            "mean_accuracy": np.mean(scores['test_accuracy']),
            "std_accuracy": np.std(scores['test_accuracy']),
            "mean_precision": np.mean(scores['test_precision_macro']),
            "mean_recall": np.mean(scores['test_recall_macro']),
            "mean_f1": np.mean(scores['test_f1_macro']),
        }

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(clean_texts).toarray()
    results[name]["TF-IDF"] = evaluate_model(X_tfidf, "TF-IDF")

    # TF-IDF + PCA
    X_tfidf_pca = PCA(n_components=100).fit_transform(X_tfidf)
    results[name]["TF-IDF + PCA"] = evaluate_model(X_tfidf_pca, "TF-IDF + PCA")

    # טוקניזציה ל-Word2Vec/FastText
    tokenized = [word_tokenize(t) for t in clean_texts]

    # Word2Vec
    w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    X_w2v = get_sentence_vectors(clean_texts, w2v_model, 100)
    results[name]["Word2Vec"] = evaluate_model(X_w2v, "Word2Vec")

    # Word2Vec + PCA
    X_w2v_pca = PCA(n_components=50).fit_transform(X_w2v)
    results[name]["Word2Vec + PCA"] = evaluate_model(X_w2v_pca, "Word2Vec + PCA")

    # FastText
    ft_model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
    X_ft = get_sentence_vectors(clean_texts, ft_model, 100)
    results[name]["FastText"] = evaluate_model(X_ft, "FastText")

    # FastText + PCA
    X_ft_pca = PCA(n_components=50).fit_transform(X_ft)
    results[name]["FastText + PCA"] = evaluate_model(X_ft_pca, "FastText + PCA")

# שמירת תוצאות
output_path = "svm_cv_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n✅ Cross-Validation results saved to '{output_path}'")
