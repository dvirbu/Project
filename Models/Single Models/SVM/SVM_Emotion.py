# מודל SVM עם ניתוח רגישות
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize

# נתיבי קבצים
fakenews_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_fakenewsnet_balanced.csv"
politifact_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB\preprocessed_politifact_balanced.csv"
output_json_file = 'svm_emotions_cv_results.json'
output_csv_file = 'svm_emotions_cv_results.csv'

# פונקציות עזר
def get_averaged_embeddings(texts_series, model_obj, dim):
    tokenized_texts = [word_tokenize(text) for text in texts_series.astype(str)]
    vectors = []
    for tokens in tokenized_texts:
        word_vecs = [model_obj.wv[token] for token in tokens if token in model_obj.wv]
        vectors.append(np.mean(word_vecs, axis=0) if word_vecs else np.zeros(dim))
    return np.array(vectors)

def process_emotion_features(emotions_column, all_possible_emotions):
    cat = pd.Categorical(emotions_column, categories=all_possible_emotions)
    return pd.get_dummies(cat, prefix='emotion')

def evaluate_model_cv(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }

def get_svm_model(random_state_val=42):
    return SVC(kernel='linear', probability=True, random_state=random_state_val)

# רגשות אפשריים
ALL_POSSIBLE_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# טעינת נתונים
try:
    fakenews_df = pd.read_csv(fakenews_path)
    politifact_df = pd.read_csv(politifact_path)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit(1)

datasets = {
    "FakeNewsNet": fakenews_df,
    "PolitiFact": politifact_df
}

final_results = {}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for dataset_name, df in datasets.items():
    df_clean = df.dropna(subset=['clean_text', 'label', 'emotion']).copy()
    texts_full = df_clean["clean_text"].astype(str)
    labels_full = df_clean["label"]
    emotions_full = df_clean["emotion"]
    emotions_ohe_full = process_emotion_features(emotions_full, ALL_POSSIBLE_EMOTIONS)

    config_scores_dataset = {
        "TF-IDF + Emotions": [],
        "TF-IDF + Emotions + PCA": [],
        "Word2Vec + Emotions": [],
        "Word2Vec + Emotions + PCA": [],
        "FastText + Emotions": [],
        "FastText + Emotions + PCA": []
    }

    print(f"\n Running Cross-Validation for {dataset_name} ")

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts_full.values, labels_full.values)):
        print(f"\nFold {fold + 1}/3")

        X_train_texts = texts_full.iloc[train_idx]
        X_test_texts = texts_full.iloc[test_idx]
        y_train = labels_full.iloc[train_idx]
        y_test = labels_full.iloc[test_idx]
        X_train_emotions = emotions_ohe_full.iloc[train_idx].values
        X_test_emotions = emotions_ohe_full.iloc[test_idx].values

        # TF-IDF
        tfidf = TfidfVectorizer(max_features=1000)
        X_train_tfidf = tfidf.fit_transform(X_train_texts.tolist()).toarray()
        X_test_tfidf = tfidf.transform(X_test_texts.tolist()).toarray()
        X_train = np.hstack((X_train_tfidf, X_train_emotions))
        X_test = np.hstack((X_test_tfidf, X_test_emotions))
        model = get_svm_model()
        config_scores_dataset["TF-IDF + Emotions"].append(
            evaluate_model_cv(X_train, X_test, y_train, y_test, model)
        )

        # TF-IDF + PCA
        pca = PCA(n_components=100, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        config_scores_dataset["TF-IDF + Emotions + PCA"].append(
            evaluate_model_cv(X_train_pca, X_test_pca, y_train, y_test, get_svm_model())
        )

        # Word2Vec
        w2v = Word2Vec(sentences=[word_tokenize(t) for t in X_train_texts.tolist()],
                       vector_size=100, window=5, min_count=1, workers=4)
        X_train_w2v = get_averaged_embeddings(X_train_texts, w2v, 100)
        X_test_w2v = get_averaged_embeddings(X_test_texts, w2v, 100)
        scaler = StandardScaler()
        X_train_w2v_scaled = scaler.fit_transform(X_train_w2v)
        X_test_w2v_scaled = scaler.transform(X_test_w2v)
        X_train = np.hstack((X_train_w2v_scaled, X_train_emotions))
        X_test = np.hstack((X_test_w2v_scaled, X_test_emotions))
        config_scores_dataset["Word2Vec + Emotions"].append(
            evaluate_model_cv(X_train, X_test, y_train, y_test, get_svm_model())
        )

        # Word2Vec + PCA
        pca = PCA(n_components=50, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        config_scores_dataset["Word2Vec + Emotions + PCA"].append(
            evaluate_model_cv(X_train_pca, X_test_pca, y_train, y_test, get_svm_model())
        )

        # FastText
        ft = FastText(sentences=[word_tokenize(t) for t in X_train_texts.tolist()],
                      vector_size=100, window=5, min_count=1, workers=4)
        X_train_ft = get_averaged_embeddings(X_train_texts, ft, 100)
        X_test_ft = get_averaged_embeddings(X_test_texts, ft, 100)
        scaler = StandardScaler()
        X_train_ft_scaled = scaler.fit_transform(X_train_ft)
        X_test_ft_scaled = scaler.transform(X_test_ft)
        X_train = np.hstack((X_train_ft_scaled, X_train_emotions))
        X_test = np.hstack((X_test_ft_scaled, X_test_emotions))
        config_scores_dataset["FastText + Emotions"].append(
            evaluate_model_cv(X_train, X_test, y_train, y_test, get_svm_model())
        )

        # FastText + PCA
        pca = PCA(n_components=50, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        config_scores_dataset["FastText + Emotions + PCA"].append(
            evaluate_model_cv(X_train_pca, X_test_pca, y_train, y_test, get_svm_model())
        )

    def calculate_mean_std(scores_list):
        metrics = {k: [dic[k] for dic in scores_list] for k in scores_list[0]}
        return {
            f"{k}_mean": np.mean(v) for k, v in metrics.items()
        } | {
            f"{k}_std": np.std(v) for k, v in metrics.items()
        }

    for config, scores in config_scores_dataset.items():
        final_results[f"{dataset_name} - {config}"] = calculate_mean_std(scores)

# תצוגת התוצאות ושמירה
results_df = pd.DataFrame(final_results).T
print("\n Final Results ")
print(results_df)

results_df.to_json(output_json_file, indent=4)
results_df.to_csv(output_csv_file)
print(f"\nResults saved to:\n- {output_json_file}\n- {output_csv_file}")
