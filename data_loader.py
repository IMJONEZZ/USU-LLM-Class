from typing import Tuple, Optional
import pandas as pd
from datasets import load_dataset
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def star_wars_data_loader(
    random_state: int,
    is_inference: bool = False,
    split_ratio: float = 0.8,
    dataset_name: str = "andrewkroening/Star-wars-scripts-dialogue-IV-VI",
) -> Tuple[
    Annotated[pd.DataFrame, "train_data"],
    Annotated[pd.DataFrame, "test_data"]
]:
    """Dataset loader step for Star Wars dialogue data.
    
    This step loads the Star Wars dialogue dataset and optionally splits it into
    train and test sets. It can also be configured for inference mode where it
    returns a smaller subset of the data.
    
    Args:
        random_state: Random state for reproducibility
        is_inference: If True, returns a smaller subset for inference
        split_ratio: Ratio of data to use for training (default: 0.8)
        dataset_name: Name of the HuggingFace dataset to load
        
    Returns:
        A tuple containing the training and test DataFrames
    """
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset['train'])
        
        # Log the initial data size
        logger.info(f"Loaded dataset with {len(df)} records")
        
        if is_inference:
            # For inference, take a small subset
            inference_size = int(len(df) * 0.05)  # 5% of data
            df = df.sample(inference_size, random_state=random_state)
            logger.info(f"Created inference subset with {len(df)} records")
            return df, pd.DataFrame()  # Empty DataFrame for test in inference mode
        
        # Split data into train and test sets
        train_size = int(len(df) * split_ratio)
        df = df.sample(frac=1, random_state=random_state)  # Shuffle
        
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # Reset indices
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        logger.info(f"Split dataset into {len(train_df)} training and {len(test_df)} test records")
        
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise