import pandas as pd
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reduce_cols(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Reduces columns in a DataFrame by aggregating prefixed time-series features.

    Args:
        df (pd.DataFrame): Input DataFrame containing time-series columns with prefixes.
        verbose (bool): If True, logs detailed information.

    Returns:
        pd.DataFrame: Aggregated DataFrame with reduced features.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    required_prefixes: List[str] = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity']
    target_columns: List[str] = ['bg+1:00', 'p_num', 'time']

    aggregated_df = pd.DataFrame(index=df.index)

    for prefix in required_prefixes:
        # Get all columns that start with the prefix
        prefix_cols = [col for col in df.columns if col.startswith(prefix)]
        if not prefix_cols:
            if verbose:
                logger.warning(f"No columns found for prefix '{prefix}'")
            continue
        if prefix == 'activity':
            try:
                mode_series = df[prefix_cols].mode(axis=1)
                aggregated_df['activity'] = mode_series.iloc[:, 0] if not mode_series.empty else pd.NA
            except Exception as e:
                logger.error(f"Error computing mode for '{prefix}': {e}")
                aggregated_df['activity'] = pd.NA
        else:
            try:
                aggregated_df[prefix] = df[prefix_cols].mean(axis=1, skipna=True)
            except Exception as e:
                logger.error(f"Error computing mean for '{prefix}': {e}")
                aggregated_df[prefix] = pd.NA

    for col in target_columns:
        if col in df.columns:
            aggregated_df[col] = df[col]
        else:
            if verbose:
                logger.info(f"Optional column '{col}' not found in input.")

    return aggregated_df
