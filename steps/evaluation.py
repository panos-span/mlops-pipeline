import logging
import pandas as pd
from zenml import step


@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluate the model.
    Args:
        df: the cleaned data
    """
    try:
        logging.info("Evaluating the model")
        # Evaluate the model
        pass
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
