import joblib
import numpy as np
import logging
from typing import List, Union

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str):
    """
    Load a trained model from disk using joblib.
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from '{model_path}'")
        return model
    except FileNotFoundError:
        logger.error(f"Model file '{model_path}' not found.")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def validate_input(data: Union[List[float], np.ndarray], expected_features: int) -> np.ndarray:
    """
    Validates and reshapes input data for prediction.
    """
    if isinstance(data, list):
        data = np.array(data)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, but got {data.shape[1]}")

    return data

def predict(input_data: Union[List[float], np.ndarray], expected_features: int):
    """
    Makes a prediction using the loaded model and validated input.
    """
    try:
        model = load_model('Models/xgboost.pkl')
        input_array = validate_input(input_data, expected_features)
        predictions = model.predict(input_array)
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
