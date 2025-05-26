import joblib
from typing import Dict,List
from sklearn.preprocessing import LabelEncoder
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_encoders(filepath: str) -> Dict[str, LabelEncoder]:
    """
    Load a previously saved encoders dictionary.

    Args:
        filepath (str): Path where encoders were dumped.

    Returns:
        Dict[str, LabelEncoder]
    """
    try:
        encoders = joblib.load(filepath)
        logger.info(f"Encoders loaded from '{filepath}'.")
        return encoders
    except FileNotFoundError:
        logger.error(f"Encoders file '{filepath}' not found.")
        raise
    except Exception as e:
        logger.error(f"Failed to load encoders from '{filepath}': {e}")
        raise

def transform_with_encoders(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Apply a pre‐fitted set of LabelEncoders to a new DataFrame (inference time).

    Args:
        df (pd.DataFrame): New data containing the same categorical columns.
        encoders (Dict[str, LabelEncoder]): Mapping column → fitted LabelEncoder.
        verbose (bool): If True, logs missing columns or transform failures.

    Returns:
        pd.DataFrame: Copy of df with categorical columns transformed.
    """
    out = df.copy()
    for col, le in encoders.items():
        if col not in out.columns:
            if verbose:
                logger.warning(f"Column '{col}' not in new DataFrame; skipping transform.")
            continue
        try:
            logger.info(f"Classes in encoders: {print(le.classes_)}")
            out[col] = le.transform(out[col].astype(str))
        except Exception as e:
            logger.error(f"Failed to transform column '{col}': {e}")
            raise
    return out