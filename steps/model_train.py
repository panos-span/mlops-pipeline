import logging

import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    """
    Train the model.
    Args:
        df: the cleaned data
    """
    try:
        logging.info("Training the model")
        # Train the model
        pass
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e