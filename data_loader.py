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
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    dataset_name: str = "andrewkroening/Star-wars-scripts-dialogue-IV-VI",
) -> Tuple[
    Annotated[pd.DataFrame, "train_data"],
    Annotated[pd.DataFrame, "validation_data"],
    Annotated[pd.DataFrame, "test_data"]
]:
    """Dataset loader step for Star Wars dialogue data.
    
    This step loads the Star Wars dialogue dataset and splits it into
    train, validation, and test sets. It can also be configured for inference mode 
    where it returns a smaller subset of the data.
    
    Args:
        random_state: Random state for reproducibility
        is_inference: If True, returns a smaller subset for inference
        train_ratio: Ratio of data to use for training (default: 0.7)
        validation_ratio: Ratio of data to use for validation (default: 0.15)
        dataset_name: Name of the HuggingFace dataset to load
        
    Returns:
        A tuple containing the training, validation, and test DataFrames
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
            # Return the inference set and empty DataFrames for validation and test
            return df, pd.DataFrame(), pd.DataFrame()
        
        # Calculate split sizes
        total_size = len(df)
        train_size = int(total_size * train_ratio)
        validation_size = int(total_size * validation_ratio)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=random_state)
        
        # Split into train, validation, and test sets
        train_df = df[:train_size]
        validation_df = df[train_size:train_size + validation_size]
        test_df = df[train_size + validation_size:]
        
        # Reset indices
        train_df = train_df.reset_index(drop=True)
        validation_df = validation_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        logger.info(
            f"Split dataset into {len(train_df)} training, "
            f"{len(validation_df)} validation, and "
            f"{len(test_df)} test records"
        )
        
        return train_df, validation_df, test_df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise