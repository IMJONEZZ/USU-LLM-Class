from pipeline import star_wars_pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)

def main():
    """Run the Star Wars dialogue processing pipeline."""
    try:
        logger.info("Starting Star Wars dialogue processing pipeline...")
        
        # Run the pipeline with default parameters
        pipeline_instance = star_wars_pipeline()
        
        logger.info("Pipeline execution completed successfully!")
        return pipeline_instance
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()