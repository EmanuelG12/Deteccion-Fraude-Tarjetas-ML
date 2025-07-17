# models/train_logistic_regression_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# CONSTANTES
RANDOM_STATE=2018

def load_data():
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')

    X_train = train.drop('Class', axis=1)
    y_train = train['Class']

    X_test = test.drop('Class', axis=1)
    y_test = test['Class']

    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename='artifacts/logistic_regression.pkl'):
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, filename)
    print(f'Model saved to {filename}')

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_logistic_regression(X_train, y_train)
    save_model(model)

if __name__ == '__main__':
    main()
