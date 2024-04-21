import os
from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        try:
            # Base directory containing JSON files
            data_dir = Path(self.config.data_dir) 

            # Create Train and Test subfolders
            train_folder = data_dir / "Train"
            test_folder = data_dir / "Test"

            # Ensure Train and Test folders exist
            train_folder.mkdir(exist_ok=True)
            test_folder.mkdir(exist_ok=True)

            # Get list of all JSON files in data_dir
            json_files = [f for f in data_dir.iterdir() if f.suffix == ".json"]

            # Iterate through each JSON file
            for json_file in json_files:
                # Load data from JSON file
                data = pd.read_json(json_file, lines=True)
                
                # Split the data into training and test sets (75% train, 25% test)
                train, test = train_test_split(data, test_size=0.25, random_state=42)

                # Define file paths for train and test JSON files in respective folders
                train_file = train_folder / f"{json_file.stem}_train.json"
                test_file = test_folder / f"{json_file.stem}_test.json"

                # Save the train and test sets to JSON files
                train.to_json(train_file, orient='records', lines=True)
                test.to_json(test_file, orient='records', lines=True)

                # Logging information
                logger.info(f"Split {json_file.name} into train and test sets")
                logger.info(f"Training set shape: {train.shape}")
                logger.info(f"Test set shape: {test.shape}")

                # Print the shapes of the training and test sets
                print(f"Training set shape for {json_file.name}: {train.shape}")
                print(f"Test set shape for {json_file.name}: {test.shape}")

        except Exception as e:
            logger.error("Error during train/test split: ", exc_info=True)
            raise  # Re-raise the exception after logging