import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class defining strategy for data handling.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Abstract method to handle data.
        Args:
            data: the data to be handled
        Returns:
            Union[pd.DataFrame, None]: the handled data
        """
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Strategy to preprocess data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data.
        Args:
            data: the data to be preprocessed
        Returns:
            pd.DataFrame: the preprocessed data
        """
        try:
            # For simplicity, we will just drop columns
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ], axis=1
            )
            # For simplicity, we will fill the nan values with the median
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review",
                                                  inplace=True)  # Could add encoding/tokenization strategy here
            # Get only numerical columns to keep it simple
            data = data.select_dtypes(include=[np.number])
            # Drop columns that are not needed
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e


class TokenizeStrategy(DataStrategy):
    """
    Strategy to tokenize data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.Series:
        """
        Tokenize the data.
        Args:
            data: the data to be tokenized
        Returns:
            pd.Series: the tokenized data
        """
        try:
            # Tokenize the review_comment_message column with Hugging Face's tokenizers
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
            data["review_comment_message"] = data["review_comment_message"].apply(
                lambda x: tokenizer.encode(x, add_special_tokens=True)
            )
            return data["review_comment_message"]
        except Exception as e:
            logging.error(f"Error in tokenizing data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for diving data into training and testing sets.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide the data into training and testing sets.
        Args:
            data: the data to be divided
        Returns:
            Union[pd.DataFrame, pd.Series]: the divided data
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e


class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into training and testing sets.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        """
        Constructor for the DataCleaning class.
        Args:
            data: the data to be cleaned
            strategy: the strategy to be used for data cleaning
        """
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle the data using the strategy.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e

