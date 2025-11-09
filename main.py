import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

# Placeholder path for dataset: data/netflix_titles.csv
DATA_PATH = os.path.join('data','netflix_titles.csv')

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        print(f'WARNING: dataset not found at {path}. Please put netflix_titles.csv in the data/ folder.')
        return None
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()
    # Keep a few useful columns and simple cleaning
    cols = ['type','title','release_year','duration','listed_in','country']
    df = df[[c for c in cols if c in df.columns]]
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float, errors='ignore')
    df['country'] = df['country'].fillna('Unknown')
    df['listed_in'] = df['listed_in'].fillna('Unknown')
    # Simplify task: predict type (Movie or TV Show)
    df = df.dropna(subset=['type','release_year','duration_num'])
    # Encode categorical features
    le_country = LabelEncoder()
    df['country_enc'] = le_country.fit_transform(df['country'].astype(str))
    le_listed = LabelEncoder()
    df['listed_enc'] = le_listed.fit_transform(df['listed_in'].astype(str))
    X = df[['release_year','duration_num','country_enc','listed_enc']]
    y = le_country = LabelEncoder().fit_transform(df['type'])
    return X, y

def train_and_evaluate(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test,preds))
    print(classification_report(y_test,preds))

def main():
    df = load_data()
    if df is None:
        return
    X,y = preprocess(df)
    train_and_evaluate(X,y)

if __name__ == '__main__':
    main()
