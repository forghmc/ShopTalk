import os
import json
import gzip
from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # Function to concatenate all values in nested JSON objects
    def concatenate_nested_values(self,nested_obj):
        return " ".join([str(v) for v in nested_obj.values()])
    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model
    # Helper function to process JSON files from a given folder and output them to a destination folder
    def process_json_files(self, source_folder, output_folder):
        processed_data = []
        # List of keys to check in each JSON object
        keys_to_check = ["item_id", "product_type", "brand", "model_name", "item_name_in_en_us", "bullet_point", "color", "style", "main_image_id", "item_keywords"]
        # Get all JSON files in the source folder
        json_files = [f for f in os.listdir(source_folder) if f.endswith(".json")]

        for json_file in json_files:
            file_path = os.path.join(source_folder, json_file)

            # Read the file line by line to avoid JSON decoding errors
            with open(file_path, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    try:
                        json_obj = json.loads(line)  # Load the JSON object from the line
                        new_obj = {}
                        for key in keys_to_check:
                            if key in json_obj:
                                value = json_obj[key]
                                if isinstance(value, list):
                                    # If it's a list of nested JSON objects, concatenate their values
                                    concatenated_string = " ".join([self.concatenate_nested_values(v) for v in value])
                                    new_obj[key] = concatenated_string
                                elif isinstance(value, dict):
                                    # If it's a single nested JSON object, concatenate its values
                                    new_obj[key] = self.concatenate_nested_values(value)
                                else:
                                    # If it's a simple value, store it directly
                                    new_obj[key] = value
                        processed_data.append(new_obj)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {line}")

        # Output the processed data to a new file in the output folder
        output_file_path = os.path.join(output_folder, f"processed_{os.path.basename(source_folder)}.csv")
        # Create a DataFrame from the processed data
        df = pd.DataFrame(processed_data)

        # Write the DataFrame to a CSV file
        df.to_csv(output_file_path, index=False)

        return output_file_path

    # I am only adding train_test_spliting cz this data is already cleaned up
    def feature_extraction(self):
        data_dir = Path(self.config.data_dir) 

        # Create Train and Test subfolders
        train_folder_path = data_dir / "Train"
        test_folder_path = data_dir / "Test"

        # Check if the Test and Train folders exist
        if not os.path.exists(test_folder_path):
            raise FileNotFoundError(f"Test folder not found: {test_folder_path}")

        if not os.path.exists(train_folder_path):
            raise FileNotFoundError(f"Train folder not found: {train_folder_path}")

        # Destination folder for output files
        destination_folder = self.config.root_dir

        # Create Test and Train subfolders in the destination folder
        os.makedirs(os.path.join(destination_folder, "Test"), exist_ok=True)
        os.makedirs(os.path.join(destination_folder, "Train"), exist_ok=True)

        # Process the Test and Train folders and store the results in respective subfolders
        test_output_path = self.process_json_files(test_folder_path, os.path.join(destination_folder, "Test"))
        train_output_path = self.process_json_files(train_folder_path, os.path.join(destination_folder, "Train"))

        # Return the paths to the created output files
        test_output_path, train_output_path  # These indicate where the files are stored

        metadata_path = os.path.join(self.config.metadata_dir, "images.csv.gz")
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
        test_csv_path_df = pd.read_csv(test_output_path)
        train_csv_path_df = pd.read_csv(train_output_path)

        image_dataset_test = test_csv_path_df.merge(image_metadata_df, left_on="main_image_id", right_on="image_id")
        image_dataset_train = train_csv_path_df.merge(image_metadata_df, left_on="main_image_id", right_on="image_id")
        image_dataset_train['combined'] = image_dataset_train.astype(str).apply(lambda row: ' '.join(row.values), axis=1)
        image_dataset_test['combined'] = image_dataset_test.astype(str).apply(lambda row: ' '.join(row.values), axis=1)
        
         # Output the processed data to a new file in the output folder
        test_output_path = os.path.join(destination_folder, f"Merged_Test{os.path.basename(destination_folder)}.csv")
        train_output_path = os.path.join(destination_folder, f"Merged_Train{os.path.basename(destination_folder)}.csv")
    
        image_dataset_test.to_csv(test_output_path)
        image_dataset_train.to_csv(train_output_path)

  
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
    
    def cleanup_imagecaptioncsv(self):
        try:
            # Path to the original CSV file
            data_dir = Path(self.config.ingest_dir) 
            csv_file_path = os.path.join(data_dir, "processed_dataset_target_data_with_captions_only.csv")

            # Define the columns to keep
            columns_to_keep = ['item_id', 'captions']

            # Load the CSV data into a DataFrame, keeping only the specified columns
            df = pd.read_csv(csv_file_path, usecols=columns_to_keep)

            # Save the updated DataFrame to the new CSV file, without the index
            df.to_csv(csv_file_path, index=False)

            # Destination folder for output files
            destination_folder = Path(self.config.root_dir)

            test_csv_path = os.path.join(destination_folder, "Merged_Testdata_validation.csv")
            train_csv_path = os.path.join(destination_folder, "Merged_Traindata_validation.csv")

            # Output the processed data to a new file in the output folder
            test_output_path = os.path.join(destination_folder, f"Merged_Test{os.path.basename(destination_folder)}.csv")
            train_output_path = os.path.join(destination_folder, f"Merged_Train{os.path.basename(destination_folder)}.csv")

            # Read the CSV file into a DataFrame
            csv_file_path_df = pd.read_csv(csv_file_path)
            test_csv_path_df = pd.read_csv(test_output_path)
            train_csv_path_df = pd.read_csv(train_output_path)

            image_dataset_test = test_csv_path_df.merge(csv_file_path_df, left_on="item_id", right_on="item_id")
            image_dataset_train = train_csv_path_df.merge(csv_file_path_df, left_on="item_id", right_on="item_id")
            
            # Merge 'combined' and 'captions' columns, prioritizing non-empty 'captions'
            image_dataset_test['combined'] = image_dataset_test['combined'].fillna('') + image_dataset_test['captions']
            image_dataset_train['combined'] = image_dataset_train['combined'].fillna('') + image_dataset_train['captions']

            # Drop 'captions' as it is now merged
            #image_dataset_test.drop(columns=['captions'], inplace=True)
            #image_dataset_train.drop(columns=['captions'], inplace=True)

            image_dataset_test.to_csv(test_output_path)
            image_dataset_train.to_csv(train_output_path)
        except Exception as e:
            logger.error("Error during train/test split: ", exc_info=True)
            raise  # Re-raise the exception after logging
    
 
########################################## / TO DO Test Case to check if the main_image_d and imag_id based merge happened properly ####################################
########################################## / TO DO Test Case to check if the string concatenation in combined column happend correctly, that can be checked again all columns using spaces from combined as delimiter####################################