import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC

base_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB"
fakenewsnet_path = os.path.join(base_path, "preprocessed_fakenewsnet_balanced.csv")
politifact_path = os.path.join(base_path, "preprocessed_politifact_balanced.csv")
output_csv_path = os.path.join(base_path, "emotion_configuration_analysis_results_onehot.csv")

try:
    fakenewsnet_df = pd.read_csv(fakenewsnet_path)
    print(f" Loaded: {fakenewsnet_path}, Rows: {len(fakenewsnet_df)}")
    politifact_df = pd.read_csv(politifact_path)
    print(f" Loaded: {politifact_path}, Rows: {len(politifact_df)}")
except FileNotFoundError as e:
    print(f" File not found: {e.filename}")
    sys.exit()

datasets = {
    'fakenewsnet': fakenewsnet_df,
    'politifact': politifact_df
}

# קונפיגורציות
emotions = ['joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral']
configurations = {
    **{emotion: [emotion] for emotion in emotions},  # כל רגש בנפרד
    'positive_emotions': ['joy', 'surprise'],
    'negative_emotions': ['sadness', 'anger', 'disgust', 'fear'],
    'all_emotions_excluding_neutral': ['joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear'],
    'all_emotions_all': emotions
}

# הגדרת המודל
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='linear', probability=True, random_state=42))
]

results = []

# הרצה
for dataset_name, df in datasets.items():
    print(f"\n Dataset: {dataset_name}")

    for config_name, emotion_list in configurations.items():
        print(f" Config: {config_name} -> {emotion_list}")

        filtered_df = df[df['emotion'].isin(emotion_list)].copy()
        if len(filtered_df) < 30:
            print(f" Skipping {config_name} (not enough rows)")
            continue

        if filtered_df['label'].nunique() < 2:
            print(f" Skipping {config_name} (not enough classes)")
            continue

        emotion_onehot = pd.get_dummies(filtered_df['emotion'])

        # יצירת מאפייני טקסט
        vectorizer = TfidfVectorizer(max_features=5000)
        X_text = vectorizer.fit_transform(filtered_df['clean_text']).toarray()

        # איחוד בין טקסט לרגש
        X = np.hstack([X_text, emotion_onehot.values])
        y = filtered_df['label'].values

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        y_preds = []
        y_tests = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = StackingClassifier(
                estimators=estimators,
                final_estimator=RandomForestClassifier(n_estimators=10, random_state=42)
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_preds.extend(y_pred)
            y_tests.extend(y_test)

        report = classification_report(y_tests, y_preds, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_tests, y_preds)

        results.append({
            'Dataset': dataset_name,
            'Configuration': config_name,
            'Accuracy': accuracy,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score'],
            'Support': report['weighted avg']['support']
        })

results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)
print(f"\n✅ Results saved to: {output_csv_path}")
