import pandas as pd
import joblib
import os

def load_data(path):
    return pd.read_csv(path, encoding="latin-1")

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

