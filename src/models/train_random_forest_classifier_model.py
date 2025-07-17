# model/train_random_forest_classifier_model.py

import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# CONSTANTES
RANDOM_STATE = 2018
RFC_METRIC = 'gini'  # Metrica usada para RandomForestClassifier
NUM_ESTIMATORS = 100 # Numeros de estimadores para RandomForestClassifier
NO_JOBS = 4 # Numeros de jobs paralelos para RandomForestClassifier

def load_data():
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')

    X_train = train.drop('Class', axis=1)
    y_train = train['Class']

    X_test = test.drop('Class', axis=1)
    y_test = test['Class']

    return X_train, X_test, y_train, y_test

def train_random_forest_classifier(X_train,y_train): 
    model = RandomForestClassifier(random_state=RANDOM_STATE,
                                   criterion=RFC_METRIC,
                                   n_estimators=NUM_ESTIMATORS,
                                   verbose=False)
    model.fit(X_train, y_train)

    return model

def save_model(model, filename='artifacts/random_forest_classifier.pkl'):
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model, filename)
    print(f'Model saved to {filename}')

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_random_forest_classifier(X_train, y_train)
    save_model(model)

if __name__ == '__main__':
    main()