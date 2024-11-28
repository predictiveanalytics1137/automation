from sklearn.experimental import enable_iterative_imputer  # Needed to enable IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
import pandas as pd
import logging
from src.logging_config import get_logger

logger = get_logger(__name__)




# def handle_categorical_features(df, target_column=None, cardinality_threshold=10):
#     """
#     Automatically handles categorical features in a DataFrame based on the number of unique categories.
    
#     Parameters:
#     - df: pandas DataFrame containing the dataset
#     - target_column: string, name of the target column if available (for Target Encoding)
#     - cardinality_threshold: integer, the threshold to apply encoding strategies.
    
#     Returns:
#     - df_encoded: pandas DataFrame with encoded categorical features
#     """
#     logger.info("Starting to handle categorical features...")
#     try:
#         # Identify categorical columns
#         categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
#         logger.info(f"Categorical columns identified: {categorical_columns}")

#         df_encoded = df.copy()
#         ordinal_encoder = OrdinalEncoder()
#         target_encoder = TargetEncoder()

#         for column in categorical_columns:
#             unique_values = df[column].nunique()
#             logger.info(f"Processing column '{column}' with {unique_values} unique values.")

#             if column.lower().endswith('_rank') or column.lower().startswith('level'):
#                 logger.info(f"Applying Ordinal Encoding to column '{column}'.")
#                 df_encoded[column] = ordinal_encoder.fit_transform(df[[column]])

#             elif unique_values <= cardinality_threshold:
#                 logger.info(f"Applying One-Hot Encoding to column '{column}'.")
#                 one_hot_encoded = pd.get_dummies(df[column], prefix=column)
#                 df_encoded = pd.concat([df_encoded.drop(column, axis=1), one_hot_encoded], axis=1)

#             elif target_column and target_column in df.columns:
#                 logger.info(f"Applying Target Encoding to column '{column}'.")
#                 df_encoded[column] = target_encoder.fit_transform(df[column], df[target_column])

#         logger.info("Completed handling of categorical features.")
#         return df_encoded

#     except Exception as e:
#         logger.error(f"Error while handling categorical features: {e}")
#         raise



logger = logging.getLogger(__name__)

def handle_categorical_features(df, target_column=None, cardinality_threshold=10):
    """
    Automatically handles categorical features in a DataFrame based on the number of unique categories.
    
    Parameters:
    - df: pandas DataFrame containing the dataset
    - target_column: string, name of the target column if available (for Target Encoding)
    - cardinality_threshold: integer, the threshold to apply encoding strategies.
    
    Returns:
    - df_encoded: pandas DataFrame with encoded categorical features
    - encoders: Dictionary of encoders used for each column
    """
    logger.info("Starting to handle categorical features...")
    try:
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Categorical columns identified: {categorical_columns}")

        df_encoded = df.copy()
        ordinal_encoder = OrdinalEncoder()
        target_encoder = TargetEncoder()
        
        encoders = {}  # Dictionary to store encoders

        for column in categorical_columns:
            unique_values = df[column].nunique()
            logger.info(f"Processing column '{column}' with {unique_values} unique values.")

            if column.lower().endswith('_rank') or column.lower().startswith('level'):
                logger.info(f"Applying Ordinal Encoding to column '{column}'.")
                df_encoded[column] = ordinal_encoder.fit_transform(df[[column]])
                encoders[column] = ordinal_encoder  # Save encoder used for the column

            elif unique_values <= cardinality_threshold:
                logger.info(f"Applying One-Hot Encoding to column '{column}'.")
                one_hot_encoded = pd.get_dummies(df[column], prefix=column)
                df_encoded = pd.concat([df_encoded.drop(column, axis=1), one_hot_encoded], axis=1)

            elif target_column and target_column in df.columns:
                logger.info(f"Applying Target Encoding to column '{column}'.")
                df_encoded[column] = target_encoder.fit_transform(df[column], df[target_column])
                encoders[column] = target_encoder  # Save encoder used for the column

        logger.info("Completed handling of categorical features.")
        
        # Save encoders for future use
        # Optionally, you can save each encoder using joblib or pickle
        # joblib.dump(encoders, 'encoders.joblib')  # Save encoders to a file

        return df_encoded, encoders

    except Exception as e:
        logger.error(f"Error while handling categorical features: {e}")
        raise
