import os
import urllib.request as request
import zipfile
import tarfile
import gzip
import time
from mlProject import logger
from mlProject.utils.common import get_size
from mlProject.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")



    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def cleanup_files(self, file_paths):
        """
        Attempts to clean up files that could not be deleted during the extraction process.
        """
        for file_path in file_paths:
            try:
                os.remove(file_path)
                print(f"File {file_path} deleted successfully during cleanup.")
            except PermissionError as e:
                print(f"Failed to delete {file_path} during cleanup: {e}")   
    
    def extract_compressed_files_from_tar(self):
        # List to keep track of files that could not be deleted
        files_to_cleanup = []
        """
        Extracts all folders/files inside 'data.tar' which could be compressed by any format.
        """
        # Path to the 'data.tar' file
        tar_file_path = Path(self.config.unzip_dir) / 'data.tar'
        
        # Create a directory to extract the contents of 'data.tar'
        tar_extract_dir = tar_file_path.parent / 'data_tar_extracted'
        os.makedirs(tar_extract_dir, exist_ok=True)
        
        # Extract the 'data.tar' file
        with tarfile.open(tar_file_path, 'r') as tar:
            tar.extractall(path=tar_extract_dir)
        
        # Walk through the directory where 'data.tar' was extracted
        for root, dirs, files in os.walk(tar_extract_dir):
            for file in files:
                file_path = Path(root) / file
                
                # Handle gzip compression
                if file_path.suffix == '.gz':
                    with gzip.open(file_path, 'rb') as f_in:
                        output_file_path = file_path.with_suffix('')
                        with open(output_file_path, 'wb') as f_out:
                            f_out.write(f_in.read())
                            try:
                                os.remove(file_path)
                            except PermissionError as e:
                                print(f"Could not delete {file_path}: {e}")
                                files_to_cleanup.append(file_path)
                # Handle zip compression
                elif file_path.suffix == '.zip':
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(path=file_path.parent)
                    os.remove(file_path)  # Remove the .zip file after extraction
        self.cleanup_files(files_to_cleanup)   