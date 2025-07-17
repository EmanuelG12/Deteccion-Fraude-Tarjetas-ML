# models/evaluate_models.py
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import shutil

def load_test_data():
    test = pd.read_csv('data/processed/test.csv')
    X_test = test.drop('Class', axis=1)
    y_test = test['Class']
    return X_test, y_test

def evaluate_model(model_path, model_name, X_test, y_test):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    if y_proba is not None:
        roc_auc = round(roc_auc_score(y_test, y_proba), 4)
        print("ROC AUC Score:", roc_auc)
    else:
        roc_auc = None

    # F1-score específico para la clase 1 (fraud)
    f1 = round(f1_score(y_test, y_pred, pos_label=1), 4)
    print("F1-Score (fraud class):", f1)

    return f1

def main():
    X_test, y_test = load_test_data()

    models = [
        ('artifacts/logistic_regression.pkl', 'Logistic Regression'),
        ('artifacts/random_forest_classifier.pkl', 'Random Forest'),
        ('artifacts/xgboost_classifier.pkl', 'XGBoost'),
    ]

    best_f1 = 0
    best_model_path = None
    best_model_name = None

    for path, name in models:
        try:
            f1 = evaluate_model(path, name, X_test, y_test)
            if f1 > best_f1:
                best_f1 = f1
                best_model_path = path
                best_model_name = name
        except Exception as e:
            print(f"❌ Error evaluando {name}: {e}")

    if best_model_path:
        shutil.copyfile(best_model_path, 'artifacts/best_model.pkl')
        print(f"\n✅ Modelo con mejor F1-score para 'fraud' ({best_f1}) guardado como 'artifacts/best_model.pkl' ({best_model_name})")

if __name__ == "__main__":
    main()
