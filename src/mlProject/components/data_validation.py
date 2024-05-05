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
           
            if not os.path.exists(self.config.root_dir):
                os.makedirs(self.config.root_dir)
            inputfile = os.path.join(self.config.input_root, os.path.basename(self.config.input_file))
            outputfile = os.path.join(self.config.root_dir, os.path.basename(self.config.output_file))
            data = pd.read_csv(inputfile)  # Adjust based on your JSON structure
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()
            # Drop columns not in schema
            cols_to_drop = all_cols - all_schema
            logger.info("Data Validation - dropping cloumns %s",cols_to_drop)

            if cols_to_drop:
                data.drop(columns=cols_to_drop, inplace=True)
                logger.info(f"Dropped columns not in schema: {cols_to_drop}")
            logger.info    
            data.to_csv(outputfile, index= False)
            logger.info("Output of data validationis phase is %s", outputfile)
            # Validate columns for each file
            file_status = all(col in all_schema for col in all_cols)
            validation_status = validation_status and file_status  # Update overall status

            # Log the validation status to a file (you could also log per file results)
            with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status for {inputfile}: {file_status}\n")

            return validation_status

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise  # Consider more specific error handling depending on your needs
