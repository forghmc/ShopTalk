import os
from mlProject import logger
from mlProject.entity.config_entity import DataValidationConfig
import pandas as pd
from pathlib import Path

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self) -> bool:
        try:
            validation_status = True  # Start assuming all is valid

            # List all json files in the directory
            files = [f for f in Path(self.config.untar_data_dir).iterdir() if f.suffix == '.json']

            for file in files:
                data = pd.read_json(file, lines=True)  # Adjust based on your JSON structure
                all_cols = list(data.columns)
                all_schema = self.config.all_schema.keys()

                # Validate columns for each file
                file_status = all(col in all_schema for col in all_cols)
                validation_status = validation_status and file_status  # Update overall status

                # Log the validation status to a file (you could also log per file results)
                with open(self.config.STATUS_FILE, 'a') as f:
                    f.write(f"Validation status for {file.name}: {file_status}\n")

            return validation_status

        except Exception as e:
            print(f"An error occurred: {e}")
            raise  # Consider more specific error handling depending on your needs
