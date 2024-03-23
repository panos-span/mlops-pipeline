import logging

import pandas as pd
from zenml import step


class IngestData:
    """
    Class to ingest data from a CSV file.
    """

    def __init__(self, data_path: str):
        """
        Constructor for the IngestData class.
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingest data from the CSV file.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingest data from a CSV file.

    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error in ingesting data: {e}")
        raise e
