
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print("sys.path:", sys.path)
from src.pipeline import clean_data


if __name__ == "__main__":
    csv_path = 'data/StudentsPerformance.csv'
    target_column = 'math_score'

    best_model, best_params = clean_data(csv_path, target_column)
    print("Training complete.")
