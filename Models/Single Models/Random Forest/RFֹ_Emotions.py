# הרצת מודל Random Forest עם ניתוח רגשות
import pandas as pd
import numpy as np
import json
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize

# נתיבי קבצים
fakenews_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv"
politifact_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv"
output_json_file = 'rf_emotions_cv_results.json'
output_csv_file = 'rf_emotions_cv_results.csv'

ALL_POSSIBLE_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
RESULTS = {}

# פונקציות עזר
def preprocess_for_w2v(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    return word_tokenize(text)

def process_emotion_features(emotions_column):
    cat = pd.Categorical(emotions_column, categories=ALL_POSSIBLE_EMOTIONS)
    return pd.get_dummies(cat, prefix='emotion').values

def vectorize_embeddings(model, tokenized_texts, dim):
    vectors = []
    for tokens in tokenized_texts:
        word_vecs = [model.wv[token] for token in tokens if token in model.wv]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(dim))
    return np.array(vectors)

def apply_pca(X, n_components):
    return PCA(n_components=n_components, random_state=42).fit_transform(X)

def run_rf_model(X, y, config_name, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = report['accuracy']
    precision = np.mean([report['0']['precision'], report['1']['precision']])
    recall = np.mean([report['0']['recall'], report['1']['recall']])
    f1 = np.mean([report['0']['f1-score'], report['1']['f1-score']])
    RESULTS[f"{dataset_name} - {config_name}"] = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

# קריאת קבצים
fakenews_df = pd.read_csv(fakenews_path)
politifact_df = pd.read_csv(politifact_path)

datasets = {
    "FakeNewsNet": fakenews_df,
    "PolitiFact": politifact_df
}

for name, df in datasets.items():
    print(f"\n🔍 Processing dataset: {name}")
    df = df.dropna(subset=['clean_text', 'label', 'emotion']).copy()
    texts = df['clean_text'].astype(str).tolist()
    labels = df['label'].tolist()
    emotions_ohe = process_emotion_features(df['emotion'])

    tokenized_texts = [preprocess_for_w2v(t) for t in texts]

    # TF-IDF + Emotions
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(texts).toarray()
    X_tfidf_emotions = np.hstack((X_tfidf, emotions_ohe))
    run_rf_model(X_tfidf_emotions, labels, "TF-IDF + Emotions", name)

    # TF-IDF + Emotions + PCA
    X_tfidf_pca = apply_pca(X_tfidf_emotions, n_components=100)
    run_rf_model(X_tfidf_pca, labels, "TF-IDF + Emotions + PCA", name)

    # Word2Vec + Emotions
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4, seed=42)
    X_w2v = vectorize_embeddings(w2v_model, tokenized_texts, 100)
    X_w2v_scaled = StandardScaler().fit_transform(X_w2v)
    X_w2v_emotions = np.hstack((X_w2v_scaled, emotions_ohe))
    run_rf_model(X_w2v_emotions, labels, "Word2Vec + Emotions", name)

    # Word2Vec + Emotions + PCA
    X_w2v_pca = apply_pca(X_w2v_emotions, n_components=50)
    run_rf_model(X_w2v_pca, labels, "Word2Vec + Emotions + PCA", name)

    # FastText + Emotions
    ft_model = FastText(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4, seed=42)
    X_ft = vectorize_embeddings(ft_model, tokenized_texts, 100)
    X_ft_scaled = StandardScaler().fit_transform(X_ft)
    X_ft_emotions = np.hstack((X_ft_scaled, emotions_ohe))
    run_rf_model(X_ft_emotions, labels, "FastText + Emotions", name)

    # FastText + Emotions + PCA
    X_ft_pca = apply_pca(X_ft_emotions, n_components=50)
    run_rf_model(X_ft_pca, labels, "FastText + Emotions + PCA", name)

# שמירת תוצאות
with open(output_json_file, "w") as f:
    json.dump(RESULTS, f, indent=4)

pd.DataFrame.from_dict(RESULTS, orient='index').to_csv(output_csv_file)

print(f"\n✅ Results saved to:\n- {output_json_file}\n- {output_csv_file}")
