from joblib import load
import pandas as pd
from src.pipeline import predict_new_data

if __name__ == "__main__":
    new_csv_path = 'data/test.csv'
    predictions = predict_new_data(new_csv_path)
    print("Predictions:")
    print(predictions)
