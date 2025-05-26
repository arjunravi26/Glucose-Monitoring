import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_null(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Handles missing values in specific columns of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        verbose (bool): If True, logs details about missing value handling.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df = df.copy()

    # Define default values and strategies
    fill_defaults = {
        'activity': 'unknown',
        'steps': 0,
        'insulin': 0,
        'carbs': 0,
        'cals': 0,
    }

    for col, default in fill_defaults.items():
        if col in df.columns:
            null_count = df[col].isna().sum()
            df[col].fillna(default, inplace=True)
            if col == 'activity':
                df[col] = df[col].astype('category')
            if verbose and null_count > 0:
                logger.info(f"Filled {null_count} nulls in '{col}' with '{default}'")
        else:
            if verbose:
                logger.warning(f"Column '{col}' not found in DataFrame")

    # Special handling for 'hr' (group-based mean)
    if 'hr' in df.columns and 'p_num' in df.columns:
        null_count = df['hr'].isna().sum()
        try:
            df['hr'].fillna(df.groupby('p_num')['hr'].transform('mean'), inplace=True)
            if verbose and null_count > 0:
                logger.info(f"Filled {null_count} nulls in 'hr' using group-wise mean per 'p_num'")
        except Exception as e:
            logger.error(f"Failed to fill 'hr' nulls with group mean: {e}")
    else:
        if verbose:
            logger.warning("'hr' or 'p_num' column missing; cannot apply group mean imputation")

    return df
