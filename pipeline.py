import pandas as pd
import logging
import argparse
from src.encoder import load_encoders, transform_with_encoders
from src.features import reduce_cols
from src.handle_null import handle_null
from src.predict import predict

# --------------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def inference_pipeline(data):
    try:
        logger.info("🔹 Reducing columns")
        data = reduce_cols(data)

        logger.info("🔹 Handling missing values")
        data = handle_null(data)

        logger.info("🔹 Loading encoders")
        encoders = load_encoders(filepath='Models/encoders.pkl')

        logger.info("🔹 Transforming data using encoders")
        data_encoded = transform_with_encoders(data, encoders)

        logger.info("🔹 Making predictions")
        prediction = predict(input_data=data_encoded, expected_features=8)

        logger.info("✅ Inference completed successfully")
        return prediction

    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")
        raise


# --------------------------------------------------------------------------------
# CLI Interface
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Values and corresponding column names
    lst = [[8.268056,	0.403264,	40.0,	86.154419,	0.0, 6.75, 'unknown', '23:55:00'],[8.268056,	0.403264,	40.0,	86.154419,	0.0, 6.75, 'unknown', '23:55:00']]
    columns = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals', 'activity', 'time']

    # Create DataFrame
    df = pd.DataFrame(lst, columns=columns)
    results = inference_pipeline(df)

    print("\n📊 Final Predictions:\n")
    print(results)
