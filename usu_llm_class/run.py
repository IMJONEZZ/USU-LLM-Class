from zenml import pipeline, step
import sys

sys.path.append(
    "C/Users/tjker/Desktop/School/Spring_2025/USU-LLM-Class/usu_llm_class/data.py"
)
from data import StarWarsDataset
from torch.utils.data import DataLoader


@step
def load_dataloader(file_path: str) -> DataLoader:
    dataset = StarWarsDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader


@pipeline
def text_processing_pipeline():
    file_path = "C:/Users/tjker/Desktop/School/Spring_2025/USU-LLM-Class/usu_llm_class/SW_EpisodeIV_VI.json"
    load_dataloader(file_path)


if __name__ == "__main__":
    run = text_processing_pipeline()
