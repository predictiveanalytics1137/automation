
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Needed to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from category_encoders import TargetEncoder
import pandas as pd
from src.logging_config import get_logger

logger = get_logger(__name__)


# def automatic_imputation(df, target_column, threshold_knn=0.05, threshold_iterative=0.15):
#     """
#     Automatically imputes missing values for both numerical and categorical features.
    
#     Parameters:
#     - df: pandas DataFrame containing the dataset
#     - target_column: string, the name of the target column (it will not be imputed)
#     - threshold_knn: float, the percentage of missing values threshold to apply KNN imputation
#     - threshold_iterative: float, the percentage of missing values threshold to apply Iterative Imputer
    
#     Returns:
#     - df: pandas DataFrame with imputed values
#     """
#     # Separate numerical and categorical columns
#     numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

#     # Remove the target column from the lists (if it's in there)
#     if target_column in numerical_columns:
#         numerical_columns.remove(target_column)
#     if target_column in categorical_columns:
#         categorical_columns.remove(target_column)

#     # Ordinal encoding for categorical columns
#     ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#     if categorical_columns:
#         df[categorical_columns] = ordinal_encoder.fit_transform(df[categorical_columns])

#     # Impute missing values for each column based on threshold
#     for column in df.columns:
#         if column == target_column:
#             continue  # Skip the target column

#         missing_percentage = df[column].isnull().mean()

#         if missing_percentage == 0:
#             continue

#         if column in categorical_columns:
#             if missing_percentage < threshold_knn:
#                 df[column] = SimpleImputer(strategy='most_frequent').fit_transform(df[[column]])
#             elif missing_percentage < threshold_iterative:
#                 df[categorical_columns] = KNNImputer(n_neighbors=5).fit_transform(df[categorical_columns])
#             else:
#                 df[categorical_columns] = IterativeImputer(max_iter=10, random_state=0).fit_transform(df[categorical_columns])

#         elif column in numerical_columns:
#             if missing_percentage < threshold_knn:
#                 df[column] = SimpleImputer(strategy='median').fit_transform(df[[column]])
#             elif missing_percentage < threshold_iterative:
#                 df[numerical_columns] = KNNImputer(n_neighbors=5).fit_transform(df[numerical_columns])
#             else:
#                 df[numerical_columns] = IterativeImputer(max_iter=10, random_state=0).fit_transform(df[numerical_columns])

#     print("Imputation complete.")
#     return df



from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Needed to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from src.logging_config import get_logger

logger = get_logger(__name__)

def automatic_imputation(df, target_column, threshold_knn=0.05, threshold_iterative=0.15):
    """
    Automatically imputes missing values for both numerical and categorical features.
    
    Parameters:
    - df: pandas DataFrame containing the dataset
    - target_column: string, the name of the target column (it will not be imputed)
    - threshold_knn: float, the percentage of missing values threshold to apply KNN imputation
    - threshold_iterative: float, the percentage of missing values threshold to apply Iterative Imputer
    
    Returns:
    - df: pandas DataFrame with imputed values
    """
    try:
        logger.info("Starting automatic imputation...")
        # Separate numerical and categorical columns
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        # Remove the target column from the lists (if it's in there)
        if target_column in numerical_columns:
            numerical_columns.remove(target_column)
        if target_column in categorical_columns:
            categorical_columns.remove(target_column)

        logger.info(f"Numerical columns: {numerical_columns}")
        logger.info(f"Categorical columns: {categorical_columns}")

        # Ordinal encoding for categorical columns
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        if categorical_columns:
            logger.info("Applying ordinal encoding to categorical columns...")
            df[categorical_columns] = ordinal_encoder.fit_transform(df[categorical_columns])

        # Impute missing values for each column based on threshold
        for column in df.columns:
            if column == target_column:
                continue  # Skip the target column

            missing_percentage = df[column].isnull().mean()
            logger.info(f"Processing column '{column}' with {missing_percentage:.2%} missing values.")

            if missing_percentage == 0:
                logger.info(f"Column '{column}' has no missing values. Skipping imputation.")
                continue

            if column in categorical_columns:
                if missing_percentage < threshold_knn:
                    logger.info(f"Applying SimpleImputer (most frequent) to column '{column}'.")
                    df[column] = SimpleImputer(strategy='most_frequent').fit_transform(df[[column]])
                elif missing_percentage < threshold_iterative:
                    logger.info("Applying KNNImputer to categorical columns...")
                    df[categorical_columns] = KNNImputer(n_neighbors=5).fit_transform(df[categorical_columns])
                else:
                    logger.info("Applying IterativeImputer to categorical columns...")
                    df[categorical_columns] = IterativeImputer(max_iter=10, random_state=0).fit_transform(df[categorical_columns])

            elif column in numerical_columns:
                if missing_percentage < threshold_knn:
                    logger.info(f"Applying SimpleImputer (median) to column '{column}'.")
                    df[column] = SimpleImputer(strategy='median').fit_transform(df[[column]])
                elif missing_percentage < threshold_iterative:
                    logger.info("Applying KNNImputer to numerical columns...")
                    df[numerical_columns] = KNNImputer(n_neighbors=5).fit_transform(df[numerical_columns])
                else:
                    logger.info("Applying IterativeImputer to numerical columns...")
                    df[numerical_columns] = IterativeImputer(max_iter=10, random_state=0).fit_transform(df[numerical_columns])

        logger.info("Imputation complete.")
        return df

    except Exception as e:
        logger.error(f"Error during automatic imputation: {e}")
        raise

