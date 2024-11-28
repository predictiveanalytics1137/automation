from sklearn.preprocessing import StandardScaler
import pandas as pd
from src import feature_engineering, hyperparameter_tuning
from src.data_preprocessing import handle_categorical_features
from src.finalization import finalize_and_evaluate_model
from src.model_selection import train_test_model_selection
import joblib
from src.utils import automatic_imputation
from src.feature_engineering import feature_engineering
from src.logging_config import get_logger

logger = get_logger(__name__)

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing import handle_categorical_features
from src.feature_engineering import feature_engineering
from src.model_selection import train_test_model_selection
from src.hyperparameter_tuning import hyperparameter_tuning
from src.finalization import finalize_and_evaluate_model
from src.logging_config import get_logger

logger = get_logger(__name__)

from src.utils import automatic_imputation  # Import from utils



# def clean_data(csv_path, target_column):
#     """
#     Complete machine learning pipeline: preprocessing, feature engineering, train-test splitting,
#     hyperparameter tuning, and final model evaluation.
    
#     Parameters:
#     - csv_path: string, path to the dataset
#     - target_column: string, name of the target column
    
#     Returns:
#     - final_metrics: dictionary containing RMSE and R-squared metrics
#     """
#     try:
#         # Load the dataset
#         logger.info(f"Loading dataset from {csv_path}...")
#         df = pd.read_csv(csv_path)
#         logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
#         logger.info(f"Original {target_column} range: {df[target_column].min()} to {df[target_column].max()}")

#         # 1. Handle Missing Values, Encoding, Feature Engineering
#         logger.info("Starting missing value imputation...")
#         df = automatic_imputation(df, target_column=target_column)
#         logger.info("Missing value imputation complete.")

#         logger.info("Handling categorical features...")
#         df_encoded = handle_categorical_features(df, cardinality_threshold=10)
#         logger.info("Categorical feature encoding complete.")

#         logger.info("Performing feature engineering...")
#         df_engineered = feature_engineering(df_encoded)
#         logger.info("Feature engineering complete.")

#         # 2. Split data into train and test
#         logger.info("Splitting data into training and testing sets...")
#         best_model_name, X_train, y_train, X_test, y_test = train_test_model_selection(
#             df_engineered, target_column=target_column, task='regression'
#         )

#         # 3. Scale the target
#         logger.info("Scaling the target variable...")
#         target_scaler = StandardScaler()
#         y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
#         y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

#         # 4. Hyperparameter Tuning
#         logger.info(f"Performing hyperparameter tuning for the best model: {best_model_name}...")
#         best_model, best_params = hyperparameter_tuning(
#             best_model_name=best_model_name, X_train=X_train, y_train=y_train_scaled, X_test=X_test, y_test=y_test_scaled, task='regression'
#         )

#         # 5. Finalize and Save Model
#         logger.info("Finalizing and evaluating the model...")
#         final_metrics = finalize_and_evaluate_model(
#             best_model.__class__, best_params, X_train, y_train_scaled, X_test, y_test_scaled
#         )
#         logger.info(f"Final metrics: {final_metrics}")

#         # Save artifacts
#         logger.info("Saving the model and associated artifacts...")
#         joblib.dump(best_model, 'best_model.joblib')
#         joblib.dump(list(X_train.columns), 'saved_feature_names.joblib')
#         joblib.dump(target_scaler, 'target_scaler.joblib')
#         logger.info("Model and artifacts saved successfully.")

#         return final_metrics

#     except Exception as e:
#         logger.error(f"Error during pipeline execution: {e}")
#         raise


import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_data(csv_path, target_column):
    """
    Complete machine learning pipeline: preprocessing, feature engineering, train-test splitting,
    hyperparameter tuning, and final model evaluation.
    
    Parameters:
    - csv_path: string, path to the dataset
    - target_column: string, name of the target column
    
    Returns:
    - final_metrics: dictionary containing RMSE and R-squared metrics
    """
    try:
        # Load the dataset
        logger.info(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # 1. Handle Missing Values, Encoding, Feature Engineering
        logger.info("Starting missing value imputation...")
        df, imputers = automatic_imputation(df, target_column=target_column)
        logger.info("Missing value imputation complete.")
        
        logger.info("Handling categorical features...")
        df_encoded, encoder = handle_categorical_features(df, cardinality_threshold=10)
        logger.info("Categorical feature encoding complete.")
        
        logger.info("Performing feature engineering...")
        df_engineered, feature_transformer = feature_engineering(df_encoded)
        logger.info("Feature engineering complete.")

        # 2. Split data into train and test
        logger.info("Splitting data into training and testing sets...")
        best_model_name, X_train, y_train, X_test, y_test = train_test_model_selection(
            df_engineered, target_column=target_column, task='regression'
        )

        # 3. Scale the target
        logger.info("Scaling the target variable...")
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        # 4. Hyperparameter Tuning
        logger.info(f"Performing hyperparameter tuning for the best model: {best_model_name}...")
        best_model, best_params = hyperparameter_tuning(
            best_model_name=best_model_name, X_train=X_train, y_train=y_train_scaled, X_test=X_test, y_test=y_test_scaled, task='regression'
        )

        # 5. Finalize and Save Model
        logger.info("Finalizing and evaluating the model...")
        final_metrics = finalize_and_evaluate_model(
            best_model.__class__, best_params, X_train, y_train_scaled, X_test, y_test_scaled
        )
        logger.info(f"Final metrics: {final_metrics}")

        # Save artifacts (model, imputers, encoder, feature transformer, etc.)
        logger.info("Saving the model and associated artifacts...")
        joblib.dump(best_model, 'best_model.joblib')
        joblib.dump(imputers, 'imputers.joblib')  # Save imputers used for handling missing values
        joblib.dump(encoder, 'encoder.joblib')  # Save the categorical encoder
        joblib.dump(feature_transformer, 'feature_transformer.joblib')  # Save the feature engineering transformer
        joblib.dump(list(X_train.columns), 'saved_feature_names.joblib')  # Save the feature names
        joblib.dump(target_scaler, 'target_scaler.joblib')  # Save the target scaler
        logger.info("Model and artifacts saved successfully.")

        return final_metrics

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise



# def predict_new_data(new_csv_path):
#     """
#     Loads the trained model and saved feature names to predict on new data without a target column,
#     applying inverse scaling to predictions if target scaling was used.
    
#     Parameters:
#     - new_csv_path: string, path to the new CSV file for prediction.
    
#     Returns:
#     - predictions: array of predictions for the target column in the original scale.
#     """
#     try:
#         logger.info(f"Loading trained model and artifacts for prediction...")
#         model = joblib.load('best_model.joblib')
#         saved_feature_names = joblib.load('saved_feature_names.joblib')
#         target_scaler = joblib.load('target_scaler.joblib')
#         logger.info("Model and artifacts loaded successfully.")

#         # Load and preprocess new data
#         logger.info(f"Loading and preprocessing new data from {new_csv_path}...")
#         new_data = pd.read_csv(new_csv_path)
#         new_data_processed = automatic_imputation(new_data, target_column=None)
#         new_data_processed = handle_categorical_features(new_data_processed, cardinality_threshold=10)
#         new_data_processed = feature_engineering(new_data_processed)
#         logger.info("New data preprocessing complete.")

#         # Align features with training data
#         logger.info("Aligning features with the training data...")
#         missing_cols = [col for col in saved_feature_names if col not in new_data_processed.columns]
#         for col in missing_cols:
#             new_data_processed[col] = 0
#         new_data_processed = new_data_processed[saved_feature_names]
#         logger.info("Feature alignment complete.")

#         # Predict on the processed new data
#         logger.info("Making predictions...")
#         predictions_scaled = model.predict(new_data_processed)

#         # Inverse transform predictions to original scale
#         predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
#         logger.info("Predictions complete.")

#         return predictions

#     except Exception as e:
#         logger.error(f"Error during prediction: {e}")
#         raise






def predict_new_data(new_csv_path):
    """
    Loads the trained model and saved preprocessing artifacts to predict on new data.
    
    Parameters:
    - new_csv_path: string, path to the new CSV file for prediction.
    
    Returns:
    - predictions: array of predictions for the target column in the original scale.
    """
    try:
        # Load the trained model and preprocessing artifacts
        logger.info("Loading trained model and artifacts for prediction...")
        model = joblib.load('best_model.joblib')
        imputers = joblib.load('imputers.joblib')
        encoder = joblib.load('encoder.joblib')
        feature_transformer = joblib.load('feature_transformer.joblib')
        saved_feature_names = joblib.load('saved_feature_names.joblib')
        target_scaler = joblib.load('target_scaler.joblib')
        logger.info("Model and artifacts loaded successfully.")

        # Load new data
        logger.info(f"Loading and preprocessing new data from {new_csv_path}...")
        new_data = pd.read_csv(new_csv_path)

        # Apply imputation and encoding to new data
        new_data_processed, _ = automatic_imputation(new_data, target_column=None, imputers=imputers)
        new_data_processed, _ = handle_categorical_features(new_data_processed, cardinality_threshold=10, encoder=encoder)
        new_data_processed, _ = feature_engineering(new_data_processed, transformer=feature_transformer)
        logger.info("New data preprocessing complete.")

        # Align features with the training data
        logger.info("Aligning features with the training data...")
        missing_cols = [col for col in saved_feature_names if col not in new_data_processed.columns]
        for col in missing_cols:
            new_data_processed[col] = 0
        new_data_processed = new_data_processed[saved_feature_names]
        logger.info("Feature alignment complete.")

        # Predict on the processed new data
        logger.info("Making predictions...")
        predictions_scaled = model.predict(new_data_processed)

        # Inverse transform predictions to original scale
        predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        logger.info("Predictions complete.")

        return predictions

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise