import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import gensim
from gensim.models import Word2Vec
import numpy as np
import os
import joblib
import sys

# הגדרת הנתיבים
base_path = r"C:\Users\dvirb\Desktop\Study\C\Project\New_Project\DB"
fakenewsnet_path = os.path.join(base_path, "preprocessed_fakenewsnet_balanced.csv")
politifact_path = os.path.join(base_path, "preprocessed_politifact_balanced.csv")
results_save_path = os.path.join(base_path, "model_evaluation_results_separated.csv")

try:
    fakenewsnet_df = pd.read_csv(fakenewsnet_path)
    print(f"📊 The file {os.path.basename(fakenewsnet_path)} has been loaded successfully. Total rows: {len(fakenewsnet_df)}")
    politifact_df = pd.read_csv(politifact_path)
    print(f"📊 The file {os.path.basename(politifact_path)} has been loaded successfully. Total rows: {len(politifact_df)}")
except FileNotFoundError as e:
    print(f"❌ ERROR: A required file was not found: {e.filename}")
    sys.exit()

# הגדרת הדאטהסטים
datasets = {
    'fakenewsnet': fakenewsnet_df,
    'politifact': politifact_df
}

# הגדרת המודלים שישתמשו בסטאקינג
estimators = [
    ('svm', SVC(kernel='linear', C=1.0, probability=True, random_state=42)),
    ('xgboost', XGBClassifier(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

# הגדרת הקונפיגורציות של רגשות
configurations = {
    'positive_emotions': ['joy', 'surprise'],
    'negative_emotions': ['sadness', 'anger', 'disgust', 'fear'],
    'all_emotions_excluding_neutral': ['sadness', 'joy', 'anger', 'disgust', 'surprise', 'fear'],
    'all_emotions_all': ['sadness', 'joy', 'anger', 'disgust', 'surprise', 'fear', 'neutral']
}

# לאחסן את כל התוצאות
all_results = []

# הפעלת האנליזה עבור כל דataset
for dataset_name, df in datasets.items():
    print(f"\n Starting analysis for dataset: {dataset_name} ")

    for config_name, emotions in configurations.items():
        print(f"\n Starting configuration for: {config_name} on {dataset_name} ")

        filtered_df = df[df['emotion'].isin(emotions)].copy()

        # בדיקה אם יש מספיק נתונים
        if len(filtered_df) < 20:
            print(f" Not enough data for configuration '{config_name}' in {dataset_name}. Skipping.")
            continue

        X = filtered_df['clean_text']
        y = filtered_df['label']

        # אם אין מספיק תוויות, נוותר על הקונפיגורציה
        if y.nunique() < 2:
            print(f"️ Not enough unique labels for configuration '{config_name}' in {dataset_name}. Skipping.")
            continue

        # הפעלת Word2Vec
        sentences = [text.split() for text in X]  # חיתוך המילים עבור כל טקסט
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)

        def vectorize_text(text):
            """ פונקציה להמיר טקסט לווקטור של ממוצע המילים ב-Word2Vec """
            words = text.split()
            vector = np.zeros(100)
            count = 0
            for word in words:
                if word in model.wv:
                    vector += model.wv[word]
                    count += 1
            if count > 0:
                vector /= count
            return vector

        # המרת הטקסטים לווקטורים
        X_vec = np.array([vectorize_text(text) for text in X])

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        y_preds_kfold = []
        y_tests_kfold = []

        print(f" Starting Cross-Validation training for {config_name} on {dataset_name} with 3 folds ")
        for train_index, test_index in kf.split(X_vec):
            X_train, X_test = X_vec[train_index], X_vec[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # מודל סטאקינג עם 3 מודלים
            stacking_model = StackingClassifier(
                estimators=estimators,
                final_estimator=RandomForestClassifier(n_estimators=10, random_state=42)
            )

            stacking_model.fit(X_train, y_train)
            y_pred = stacking_model.predict(X_test)

            y_preds_kfold.extend(y_pred)
            y_tests_kfold.extend(y_test)

        print(f"✅ Cross-Validation training for {config_name} on {dataset_name} completed successfully.")

        # יצירת דוח דיוק
        report = classification_report(y_tests_kfold, y_preds_kfold, output_dict=True, zero_division=0)

        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                results_row = {
                    'Dataset': dataset_name,
                    'Configuration': config_name,
                    'Label': label,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': metrics['support']
                }
                all_results.append(results_row)

        accuracy = accuracy_score(y_tests_kfold, y_preds_kfold)
        all_results.append({
            'Dataset': dataset_name,
            'Configuration': config_name,
            'Label': 'accuracy',
            'Precision': accuracy,
            'Recall': accuracy,
            'F1-Score': accuracy,
            'Support': len(y_tests_kfold)
        })

# שמירת התוצאות לקובץ CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv(results_save_path, index=False)
print(f"\n🎉 All evaluation results have been saved successfully to the file: {results_save_path}")
