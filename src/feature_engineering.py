from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from itertools import combinations
import pandas as pd
import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)


# def feature_engineering(df):
#     logger.info("Starting feature engineering...")
#     """
#     Automatically performs feature engineering on the DataFrame, including feature creation,
#     interaction terms, normalization, and feature selection.
    
#     Parameters:
#     - df: pandas DataFrame containing the dataset
    
#     Returns:
#     - df_fe: pandas DataFrame with engineered features
#     """
#     df_fe = df.copy()
#     numerical_columns = df_fe.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     scaler = StandardScaler()

#     # Interaction Terms
#     for comb in combinations(numerical_columns, 2):
#         df_fe[f"{comb[0]}_x_{comb[1]}"] = df_fe[comb[0]] * df_fe[comb[1]]

#     # Aggregate Features (Ratios)
#     for i, col1 in enumerate(numerical_columns):
#         for col2 in numerical_columns[i + 1:]:
#             df_fe[f"{col1}_to_{col2}"] = df_fe[col1] / (df_fe[col2] + 1e-5)

#     # Normalize Numerical Features
#     df_fe[numerical_columns] = scaler.fit_transform(df_fe[numerical_columns])

#     # Log Transformation for Positively Skewed Columns
#     for col in numerical_columns:
#         if df_fe[col].min() > 0 and df_fe[col].skew() > 1:
#             df_fe[col] = np.log1p(df_fe[col])

#     # Remove Low Variance and Highly Correlated Features
#     selector = VarianceThreshold(threshold=0.01)
#     df_fe = pd.DataFrame(selector.fit_transform(df_fe), columns=df_fe.columns[selector.get_support()])

#     corr_matrix = df_fe.corr().abs()
#     upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
#     df_fe.drop(columns=to_drop, inplace=True)

#     print(f"Feature engineering complete. Final feature count: {df_fe.shape[1]}")
#     return df_fe

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from itertools import combinations
import pandas as pd
import numpy as np
from src.logging_config import get_logger

logger = get_logger(__name__)

def feature_engineering(df):
    logger.info("Starting feature engineering...")
    """
    Automatically performs feature engineering on the DataFrame, including feature creation,
    interaction terms, normalization, and feature selection.
    
    Parameters:
    - df: pandas DataFrame containing the dataset
    
    Returns:
    - df_fe: pandas DataFrame with engineered features
    """
    try:
        # Copy the dataframe to avoid modifying the original
        df_fe = df.copy()
        numerical_columns = df_fe.select_dtypes(include=['float64', 'int64']).columns.tolist()
        logger.info(f"Identified numerical columns: {numerical_columns}")

        scaler = StandardScaler()

        # Interaction Terms
        logger.info("Creating interaction terms...")
        for comb in combinations(numerical_columns, 2):
            df_fe[f"{comb[0]}_x_{comb[1]}"] = df_fe[comb[0]] * df_fe[comb[1]]

        # Aggregate Features (Ratios)
        logger.info("Creating aggregate features (ratios)...")
        for i, col1 in enumerate(numerical_columns):
            for col2 in numerical_columns[i + 1:]:
                df_fe[f"{col1}_to_{col2}"] = df_fe[col1] / (df_fe[col2] + 1e-5)

        # Normalize Numerical Features
        logger.info("Normalizing numerical features...")
        df_fe[numerical_columns] = scaler.fit_transform(df_fe[numerical_columns])

        # Log Transformation for Positively Skewed Columns
        logger.info("Applying log transformations for skewed columns...")
        for col in numerical_columns:
            if df_fe[col].min() > 0 and df_fe[col].skew() > 1:
                df_fe[col] = np.log1p(df_fe[col])

        # Remove Low Variance Features
        logger.info("Removing low variance features...")
        selector = VarianceThreshold(threshold=0.01)
        df_fe = pd.DataFrame(selector.fit_transform(df_fe), columns=df_fe.columns[selector.get_support()])

        # Remove Highly Correlated Features
        logger.info("Removing highly correlated features...")
        corr_matrix = df_fe.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
        logger.info(f"Dropping {len(to_drop)} highly correlated columns: {to_drop}")
        df_fe.drop(columns=to_drop, inplace=True)

        logger.info(f"Feature engineering complete. Final feature count: {df_fe.shape[1]}")
        return df_fe

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise
