from zenml import pipeline, step

from evaluator import evaluate_model_
from process_data import process_data
from train_model import train_model_


@step
def tokenize_data() -> dict:
    return process_data()


@step
def train_model(data: dict) -> str:
    return train_model_(process_data())


@step
def evaluate_model(data: dict, model_path: str) -> None:
    return evaluate_model_(process_data(), train_model_(process_data()))


@pipeline
def feature_engineering_pipeline() -> None:
    # Define a pipeline that connects the steps.
    data = tokenize_data()
    model_path = train_model(data)  # Train the model and get it
    evaluate_model(data, model_path)  # Pass the model path to evaluation


if __name__ == "__main__":
    feature_engineering_pipeline()
