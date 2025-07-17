# models/train_xgboost_classifier.py

import pandas as pd
from xgboost import XGBClassifier
import joblib
import os

RANDOM_STATE = 2018

def load_data():
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')

    X_train = train.drop('Class', axis=1)
    y_train = train['Class']

    X_test = test.drop('Class', axis=1)
    y_test = test['Class']

    return X_train, X_test, y_train, y_test

def train_xgboost_classfier(X_train, y_train):
    model = XGBClassifier(n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE, 
    use_label_encoder=False)

    model.fit(X_train, y_train)

    return model

def save_model(model, filename='artifacts/xgboost_classifier.pkl'):
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, filename)
    print(f'Model saved to {filename}')

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_xgboost_classfier(X_train, y_train)
    save_model(model)

if __name__ == '__main__':
    main()
