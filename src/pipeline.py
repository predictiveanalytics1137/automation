from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.data_preprocessing import handle_categorical_features
from src.finalization import finalize_and_evaluate_model
from src.model_selection import train_test_model_selection
from src.hyperparameter_tuning import hyperparameter_tuning
import joblib
from src.utils import automatic_imputation
from src.feature_engineering import feature_engineering
from src.logging_config import get_logger
import os
import featuretools as ft
from src.feature_selection import feature_selection


from src.logging_config import get_logger
from src.helper import normalize_column_names
logger = get_logger(__name__)

from src.utils import automatic_imputation  # Import from utils


def train_pipeline(csv_path, target_column):
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
        df_encoded, encoder = handle_categorical_features(df, cardinality_threshold=3)
        saved_column_names = df_encoded.columns.tolist()

       # Ensure the target column is not included in the saved column names
        if target_column in saved_column_names:
           saved_column_names.remove(target_column)
       # Save the updated column names to a file
        joblib.dump(saved_column_names, 'saved_column_names.pkl')
        logger.info("Categorical feature encoding complete.")
        
        logger.info("Performing feature engineering...")
        feature_defs_path = os.path.splitext(os.path.basename(csv_path))[0] + "_feature_defs.pkl"

        df_engineered, feature_defs = feature_engineering(
            df_encoded, target_column=target_column, training = True
        )
        df_engineered = normalize_column_names(df_engineered)
        logger.info("Feature engineering complete.")

        # 4. Perform feature selection
        logger.info("Performing feature selection...")
        df_selected, selected_features = feature_selection(
            df=df_engineered,
            target_column=target_column,
            task="regression",
            save_path='selected_features.pkl'
    )
        logger.info(f"Selected features: {selected_features}")


        # 2. Split data into train and test
        logger.info("Splitting data into training and testing sets...")
        best_model_name, X_train, y_train, X_test, y_test = train_test_model_selection(
            df_selected, target_column=target_column, task='regression'
        )

        # 3. Scale the target
        logger.info("Scaling the target variable...")
        target_scaler = StandardScaler()
        #y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        #y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        # 4. Hyperparameter Tuning
        logger.info(f"Performing hyperparameter tuning for the best model: {best_model_name}...")
        best_model, best_params = hyperparameter_tuning(
            best_model_name=best_model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, task='regression'
        )

        # 5. Finalize and Save Model
        logger.info("Finalizing and evaluating the model...")
        final_metrics = finalize_and_evaluate_model(
            best_model.__class__, best_params, X_train, y_train, X_test, y_test
        )
        logger.info(f"Final metrics: {final_metrics}")

        directory_name = os.path.splitext(os.path.basename(csv_path))[0]  # Get the file name without extension
        if not os.path.exists(directory_name):  # Check if folder doesn't exist
            os.makedirs(directory_name)  # Create the folder

        # Save artifacts (model, imputers, encoder, feature transformer, etc.) in the created directory
        logger.info("Saving the model and associated artifacts...")
        joblib.dump(best_model, os.path.join(directory_name, 'best_model.joblib'))
        joblib.dump(imputers, os.path.join(directory_name, 'imputers.joblib'))  # Save imputers used for handling missing values
        joblib.dump(encoder, os.path.join(directory_name, 'encoder.joblib'))  # Save the categorical encoder
        # joblib.dump(feature_transformer, os.path.join(directory_name, 'feature_transformer.joblib'))  # Save the feature engineering transformer
        joblib.dump(list(X_train.columns), os.path.join(directory_name, 'saved_feature_names.joblib'))  # Save the feature names
        joblib.dump(target_scaler, os.path.join(directory_name, 'target_scaler.joblib'))  # Save the target scaler
        logger.info("Model and artifacts saved successfully.")

        return final_metrics

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise


def predict_new_data(new_csv_path, feature_defs_path="StudentsPerformance_feature_defs.pkl"):
    """
    Loads the trained model and saved preprocessing artifacts to predict on new data.
    
    Parameters:
    - new_csv_path: string, path to the new CSV file for prediction.
    - feature_defs_path: string, path to the saved feature definitions.
    
    Returns:
    - predictions: array of predictions for the target column in the original scale.
    """
    try:
        # Load the trained model and preprocessing artifacts
        logger.info("Loading trained model and artifacts for prediction...")
        model = joblib.load('StudentsPerformance/best_model.joblib')
        imputers = joblib.load('StudentsPerformance/imputers.joblib')
        encoder = joblib.load('StudentsPerformance/encoder.joblib')
        saved_feature_names = joblib.load('StudentsPerformance/saved_feature_names.joblib')
        target_scaler = joblib.load('StudentsPerformance/target_scaler.joblib')
        saved_column_names = joblib.load('saved_column_names.pkl')
        selected_features = joblib.load('selected_features.pkl')
        logger.info("Model and artifacts loaded successfully.")

        # Load new data
        logger.info(f"Loading and preprocessing new data from {new_csv_path}...")
        new_data = pd.read_csv(new_csv_path)

        # Apply imputation and encoding to new data
        logger.info("Applying imputation and encoding to new data...")
        new_data_processed, _ = automatic_imputation(new_data, target_column=None, imputers=imputers)
        new_data_processed, _ = handle_categorical_features(new_data_processed, cardinality_threshold=10, encoders=encoder, saved_column_names=saved_column_names)
        logger.info("Imputation and encoding complete.")

        # Apply feature engineering using saved feature definitions
        logger.info("Applying feature engineering using saved feature definitions...")
        #feature_defs = joblib.load("featurengineering_feature_defs.pkl")
        feature_matrix, _ = feature_engineering(new_data_processed, training=False)
        feature_matrix = normalize_column_names(feature_matrix)

        logger.info("Feature engineering complete.")

        logger.info("Applying feature selection...")
        selected_feature_matrix = feature_matrix[selected_features]  # Align with selected features from training
        logger.info(f"Feature matrix reduced to selected features: {selected_feature_matrix.shape}")


        # Predict on the processed new data
        logger.info("Making predictions...")
        predictions_scaled = model.predict(selected_feature_matrix)

        # Check the shape before reshaping
        logger.info(f"Predictions before reshape: {predictions_scaled.shape}")
        #predictions_scaled = predictions_scaled.reshape(-1, 1)
        logger.info(f"Predictions after reshape: {predictions_scaled.shape}")

        # Inverse transform predictions to original scale
        #predictions = target_scaler.inverse_transform(predictions_scaled).flatten()
        #logger.info(f"Predictions after inverse transformation: {predictions[:10]}")

        logger.info("Predictions complete.")
        #return predictions
        return predictions_scaled

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

