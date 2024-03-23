from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model


@pipeline(enable_cache=True)
def training_pipeline(data_path: str):
    """
    Training pipeline to train a model.
    Args:
        data_path: path to the data
    """
    # Ingest data
    df = ingest_df(data_path)
    # Clean data
    df = clean_data(df)
    # Train model
    train_model(df)
    # Evaluate model
    evaluate_model(df)
