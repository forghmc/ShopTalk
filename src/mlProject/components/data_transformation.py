import os
import json
import re
import gzip
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
    # Clean up 'combined' column to extract relevant values and remove dictionary-like structures
    def extract_values(self, text):
        # Regular expression to find all dictionary 'value' fields
        value_pattern = re.compile(r"{.*?'value': ?'(.*?)'}")
        
        # Find all matches for 'value' fields
        extracted_values = value_pattern.findall(text)
        
        # Clean text by removing dictionary-like structures
        cleaned_text = re.sub(r"{.*?}", "", text).replace("nan", "").strip()
        
        # Combine cleaned text and extracted values
        combined_text = cleaned_text + " " + ' '.join(extracted_values)
        
        # Return the cleaned and combined result
        return combined_text
    
    def train_test_spliting(self):
        try:
            # Base directory containing JSON files
            data_dir = Path(self.config.base_dir) 

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

            # Paths to subfolders
            image_folder = os.path.join(data_dir, "data_ingestion", "abo-images-small", "images", "small")
            metadata_path = os.path.join(data_dir, "data_ingestion", "abo-images-small", "images", "metadata", "images.csv.gz")
            test_csv_path = os.path.join(data_dir, "data_validation", "Test", "processed_Test.csv")
            train_csv_path = os.path.join(data_dir, "data_validation", "Train", "processed_Train.csv")
            test_csv_path_merged = os.path.join(data_dir, "data_validation", "Test", "processed_merged_Test.csv")
            train_csv_path_merged = os.path.join(data_dir, "data_validation", "Train", "processed__merged_Train.csv")

            # Uncompress the CSV file
            uncompressed_csv_path = metadata_path.replace(".gz", "")

            # Uncompress the file and validate columns
            with gzip.open(metadata_path, 'rt') as gz_file:
                csv_content = gz_file.read()

            # Write the uncompressed content to a new file
            with open(uncompressed_csv_path, 'w') as uncompressed_file:
                uncompressed_file.write(csv_content)

            # Read the CSV file into a DataFrame
            image_metadata_df = pd.read_csv(uncompressed_csv_path)
            test_csv_path_df = pd.read_csv(test_csv_path)
            train_csv_path_df = pd.read_csv(train_csv_path)

            image_dataset_test = test_csv_path_df.merge(image_metadata_df, left_on="main_image_id", right_on="image_id")
            image_dataset_train = train_csv_path_df.merge(image_metadata_df, left_on="main_image_id", right_on="image_id")
            image_dataset_train['combined'] = image_dataset_train.astype(str).apply(lambda row: ' '.join(row.values), axis=1)
            image_dataset_test['combined'] = image_dataset_test.astype(str).apply(lambda row: ' '.join(row.values), axis=1)
            
            # Ensure 'cleaned_combined' column exists before attempting to modify it
            if 'cleaned_combined' not in image_dataset_test.columns:
                image_dataset_test['cleaned_combined'] = ""

            if 'cleaned_combined' not in image_dataset_train.columns:
                image_dataset_train['cleaned_combined'] = ""

            # Clean and update the 'combined' column
            image_dataset_test['cleaned_combined'] += " " + image_dataset_test['combined'].str.replace(r"{.*?}", "", regex=True).str.replace(r"nan", "", regex=True).str.strip()
            image_dataset_train['cleaned_combined'] += " " + image_dataset_train['combined'].str.replace(r"{.*?}", "", regex=True).str.replace(r"nan", "", regex=True).str.strip()

            # Save cleaned datasets to CSV files
            image_dataset_test.to_csv(test_csv_path_merged, index=False)
            image_dataset_train.to_csv(train_csv_path_merged, index=False)


        except Exception as e:
            logger.error("Error during train/test split: ", exc_info=True)
            raise  # Re-raise the exception after logging