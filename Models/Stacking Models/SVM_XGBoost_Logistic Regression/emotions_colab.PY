import os
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec, FastText
from nltk.tokenize import word_tokenize
from google.colab import files

uploaded = files.upload()
fakenews_path = "preprocessed_fakenewsnet_balanced.csv"
politifact_path = "preprocessed_politifact_balanced.csv"

output_json_file = 'stacking_emotions_cv_results.json'
output_csv_file = 'stacking_emotions_cv_results.csv'

# פונקציות עזר

def get_averaged_embeddings(texts_series, model_obj, dim):
    tokenized_texts = [word_tokenize(text) for text in texts_series.astype(str)]
    
    vectors = []
    for tokens in tokenized_texts:
        word_vecs = [model_obj.wv[token] for token in tokens if token in model_obj.wv]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(dim)) 
    return np.array(vectors)

def process_emotion_features(emotions_column, all_possible_emotions):
    emotions_column_categorical = pd.Categorical(emotions_column, categories=all_possible_emotions)
    ohe_emotions = pd.get_dummies(emotions_column_categorical, prefix='emotion')
    return ohe_emotions

def get_stacking_model():
    estimators = [
        ('svm', SVC(kernel='linear', probability=True, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=1)),
        ('logreg', LogisticRegression(max_iter=1000, random_state=42))
    ]
    return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42), cv=3, n_jobs=-1)

def evaluate_model_cv(X_train_fold, X_test_fold, y_train_fold, y_test_fold, model_instance):
    model_instance.fit(X_train_fold, y_train_fold)
    y_pred = model_instance.predict(X_test_fold)
    return {
        "accuracy": accuracy_score(y_test_fold, y_pred),
        "precision": precision_score(y_test_fold, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test_fold, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test_fold, y_pred, average="weighted", zero_division=0)
    }

ALL_POSSIBLE_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

try:
    fakenews_df = pd.read_csv(fakenews_path)
    politifact_df = pd.read_csv(politifact_path)
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure files are in Google Drive and paths are correct.")
    exit()

datasets = {
    "FakeNewsNet": fakenews_df,
    "PolitiFact": politifact_df
}

final_results = {}
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


for dataset_name, df in datasets.items():
    df = df.dropna(subset=['clean_text', 'label', 'emotion'])
    
    texts = df["clean_text"].astype(str)
    labels = df["label"]
    emotions = df["emotion"]

    emotions_ohe_full = process_emotion_features(emotions, ALL_POSSIBLE_EMOTIONS)

    config_scores_dataset = {
        "TF-IDF + Emotions": [],
        "TF-IDF + Emotions + PCA": [],
        "Word2Vec + Emotions": [],
        "Word2Vec + Emotions + PCA": [],
        "FastText + Emotions": [],
        "FastText + Emotions + PCA": []
    }

    print(f"\n Running Cross-Validation for {dataset_name} with {n_splits} folds ")

    for fold, (train_index, test_index) in enumerate(skf.split(texts, labels)):
        print(f"\n {dataset_name} - Fold {fold + 1}/{n_splits} ")

        X_train_fold_texts, X_test_fold_texts = texts.iloc[train_index], texts.iloc[test_index]
        y_train_fold, y_test_fold = labels.iloc[train_index], labels.iloc[test_index]

        X_train_fold_emotions_ohe = emotions_ohe_full.iloc[train_index]
        X_test_fold_emotions_ohe = emotions_ohe_full.iloc[test_index]
        
        # 1. TF-IDF + Emotions
        print(f" Fold {fold + 1}: TF-IDF + Emotions ")
        tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_fold_texts).toarray()
        X_test_tfidf = tfidf_vectorizer.transform(X_test_fold_texts).toarray()
        
        X_train_tfidf_emotions = np.hstack((X_train_tfidf, X_train_fold_emotions_ohe.values))
        X_test_tfidf_emotions = np.hstack((X_test_tfidf, X_test_fold_emotions_ohe.values))
        
        stack_model_tfidf_emotions = get_stacking_model()
        config_scores_dataset["TF-IDF + Emotions"].append(
            evaluate_model_cv(X_train_tfidf_emotions, X_test_tfidf_emotions, y_train_fold, y_test_fold, stack_model_tfidf_emotions)
        )

        # 2. TF-IDF + Emotions + PCA
        print(f" Fold {fold + 1}: TF-IDF + Emotions + PCA ")
        pca_tfidf_emotions = PCA(n_components=100, random_state=42)
        X_train_tfidf_emotions_pca = pca_tfidf_emotions.fit_transform(X_train_tfidf_emotions)
        X_test_tfidf_emotions_pca = pca_tfidf_emotions.transform(X_test_tfidf_emotions)
        
        stack_model_tfidf_emotions_pca = get_stacking_model()
        config_scores_dataset["TF-IDF + Emotions + PCA"].append(
            evaluate_model_cv(X_train_tfidf_emotions_pca, X_test_tfidf_emotions_pca, y_train_fold, y_test_fold, stack_model_tfidf_emotions_pca)
        )

        # 3. Word2Vec + Emotions
        print(f" Fold {fold + 1}: Word2Vec + Emotions ")
        w2v_model = Word2Vec(sentences=[t.split() for t in X_train_fold_texts], vector_size=100, window=5, min_count=1, workers=4) # Train on fold's train data
        X_train_w2v = get_averaged_embeddings(X_train_fold_texts, w2v_model, 100)
        X_test_w2v = get_averaged_embeddings(X_test_fold_texts, w2v_model, 100)
        
        scaler_w2v = StandardScaler()
        X_train_w2v_scaled = scaler_w2v.fit_transform(X_train_w2v)
        X_test_w2v_scaled = scaler_w2v.transform(X_test_w2v)

        X_train_w2v_emotions = np.hstack((X_train_w2v_scaled, X_train_fold_emotions_ohe.values))
        X_test_w2v_emotions = np.hstack((X_test_w2v_scaled, X_test_fold_emotions_ohe.values))
        
        stack_model_w2v_emotions = get_stacking_model()
        config_scores_dataset["Word2Vec + Emotions"].append(
            evaluate_model_cv(X_train_w2v_emotions, X_test_w2v_emotions, y_train_fold, y_test_fold, stack_model_w2v_emotions)
        )

        # 4. Word2Vec + Emotions + PCA
        print(f" Fold {fold + 1}: Word2Vec + Emotions + PCA ")
        pca_w2v_emotions = PCA(n_components=50, random_state=42)
        X_train_w2v_emotions_pca = pca_w2v_emotions.fit_transform(X_train_w2v_emotions)
        X_test_w2v_emotions_pca = pca_w2v_emotions.transform(X_test_w2v_emotions)
        
        stack_model_w2v_emotions_pca = get_stacking_model()
        config_scores_dataset["Word2Vec + Emotions + PCA"].append(
            evaluate_model_cv(X_train_w2v_emotions_pca, X_test_w2v_emotions_pca, y_train_fold, y_test_fold, stack_model_w2v_emotions_pca)
        )

        # 5. FastText + Emotions
        print(f" Fold {fold + 1}: FastText + Emotions ")
        ft_model = FastText(sentences=[t.split() for t in X_train_fold_texts], vector_size=100, window=5, min_count=1, workers=4) # Train on fold's train data
        X_train_ft = get_averaged_embeddings(X_train_fold_texts, ft_model, 100)
        X_test_ft = get_averaged_embeddings(X_test_fold_texts, ft_model, 100)
        
        scaler_ft = StandardScaler()
        X_train_ft_scaled = scaler_ft.fit_transform(X_train_ft)
        X_test_ft_scaled = scaler_ft.transform(X_test_ft)

        X_train_ft_emotions = np.hstack((X_train_ft_scaled, X_train_fold_emotions_ohe.values))
        X_test_ft_emotions = np.hstack((X_test_ft_scaled, X_test_fold_emotions_ohe.values))
        
        stack_model_ft_emotions = get_stacking_model()
        config_scores_dataset["FastText + Emotions"].append(
            evaluate_model_cv(X_train_ft_emotions, X_test_ft_emotions, y_train_fold, y_test_fold, stack_model_ft_emotions)
        )

        # 6. FastText + Emotions + PCA
        print(f" Fold {fold + 1}: FastText + Emotions + PCA ")
        pca_ft_emotions = PCA(n_components=50, random_state=42)
        X_train_ft_emotions_pca = pca_ft_emotions.fit_transform(X_train_ft_emotions)
        X_test_ft_emotions_pca = pca_ft_emotions.transform(X_test_ft_emotions)
        
        stack_model_ft_emotions_pca = get_stacking_model()
        config_scores_dataset["FastText + Emotions + PCA"].append(
            evaluate_model_cv(X_train_ft_emotions_pca, X_test_ft_emotions_pca, y_train_fold, y_test_fold, stack_model_ft_emotions_pca)
        )
    
    # חישוב ממוצע וסטיית תקן עבור כל קונפיגורציה
    def calculate_mean_std_scores(scores_list):
        if not scores_list:
            return {}
        metrics = {k: [dic[k] for dic in scores_list] for k in scores_list[0]}
        
        mean_std_scores = {}
        for metric, values in metrics.items():
            mean_std_scores[f"{metric}_mean"] = np.mean(values)
            mean_std_scores[f"{metric}_std"] = np.std(values)
        return mean_std_scores

    for config_name, scores_list in config_scores_dataset.items():
        final_results[f"{dataset_name} - {config_name}"] = calculate_mean_std_scores(scores_list)

# שמירת תוצאות והורדה
print("\n Final Cross-Validation Results ")
results_df = pd.DataFrame(final_results).T
print(results_df)

# שמירה לקובץ JSON
json_output = results_df.to_json(indent=4)
with open(output_json_file, "w") as f:
    f.write(json_output)
print(f"\nResults saved to: {output_json_file}")

# שמירה לקובץ CSV
results_df.to_csv(output_csv_file)
print(f"Results saved to: {output_csv_file}")

# הורדה אוטומטית בColab
try:
    files.download(output_json_file)
    files.download(output_csv_file)
    print("\nFiles downloaded successfully.")
except Exception as e:
    print(f"\nError downloading files: {e}. Make sure you run this in Google Colab.")