import os
import urllib.request as request
import zipfile
import tarfile
import gzip
import time
from mlProject import logger
from mlProject.utils.common import get_size
from mlProject.entity.config_entity import DataIngestionConfig
import requests
from pathlib import Path
import pandas as pd
import shutil

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    '''
    Caution for first time this function takes time as it downloads two tar 
    files(abo-listing-tar and abo-small-images.tar).
    Then it extracts json from gzip format and make dataset.
    Combines the dataset with images metadata to that helps to load images at search.
    '''
    def download_file(self):
        try: 
            if not os.path.exists(self.config.root_dir):
                os.makedirs(self.config.root_dir)
            
            data_file_df = None
            target_output = os.path.join(self.config.root_dir, os.path.basename(self.config.target_data_image_file))
            datafilepath= os.path.join(self.config.root_dir, os.path.basename(self.config.local_data_file))
            if not os.path.exists(datafilepath):
                datafile, headers = request.urlretrieve(
                    url=self.config.source_data_url,
                    filename=datafilepath
                )
                logger.info(f"{os.path.abspath(datafile)} downloaded with the following info:\n{headers}")
                datafilepath= os.path.join(self.config.root_dir, os.path.basename(datafile))
                data_file_df = self.extract_and_filter_json_gz(datafilepath)
            else:
                
                logger.info(f"Local data file '{datafilepath}' already exists. Skipping download.")
                data_file_df = self.extract_and_filter_json_gz(datafilepath)
            
            imagefilepath=os.path.join(self.config.root_dir, os.path.basename(self.config.local_image_file))

            if not os.path.exists(imagefilepath):
                imagefile, headers = request.urlretrieve(
                    url=self.config.source_image_url,
                    filename=imagefilepath
                )
                imagefilepath=os.path.join(self.config.root_dir, os.path.basename(imagefile))
                logger.info(f"{os.path.abspath(imagefilepath)} downloaded with the following info:\n{headers}")
                if data_file_df is None:
                   logger.error("Unable to extract data from JSON.gz files.")
                   return None
                data_image_df = self.extract_image_metadata_from_tar(imagefilepath, data_file_df)
                data_image_df.to_csv(target_output,index = False)
            else:

                logger.info(f"Local image file '{imagefilepath}' already exists. Skipping download.")
                if data_file_df is None:
                   logger.error("Unable to extract data from JSON.gz files.")
                   return None
                data_image_df = self.extract_image_metadata_from_tar(imagefilepath, data_file_df)
                data_image_df.to_csv(target_output, index = False)
            logger.info("Output file for data ingestion can be found at %s", os.path.abspath(self.config.target_data_image_file))
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    
    '''
    def extract_image_metadata_from_tar(self, tar_path,data_file_df):
        # Ensure the temporary directory is ready
        print(tar_path)
        image_meta = pd.read_csv(tar_path)
        image_dataset = data_file_df.merge(image_meta, left_on="main_image_id", right_on="image_id")
        return image_dataset
    '''
    

  
    def extract_value(self, x, key='value', tag=None, tag_key='language_tag'):
        if x is None or not isinstance(x, list):
            return None
        if tag:
            values = [item[key] for item in x if isinstance(item, dict) and item.get(tag_key, '').startswith(tag)]
        else:
            values = [item[key] for item in x if isinstance(item, dict) and key in item]
        return values[0] if values else None

    def concatenate_values(self, x):
        if x is None or not isinstance(x, list):
            return None
        return ', '.join([item['value'] for item in x if isinstance(item, dict)
                          and item.get('language_tag', '').startswith('en')])

    def extract_and_filter_json_gz(self,tar_path):
        extraction_path = self.config.unzip_dir
        filtered_data = []
        if not os.path.exists(extraction_path):
            os.makedirs(extraction_path)

        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(path=extraction_path)
            for root, dirs, files in os.walk(extraction_path):
                for file in files:
                    if file.endswith('.json.gz'):
                        file_path = os.path.join(root, file)
                        print(f"Processing file: {file_path}")
                        meta = pd.read_json(file_path, lines=True, compression='gzip')
                        meta['item_name_in_en'] = meta['item_name'].apply(lambda x: self.extract_value(x, tag='en'))
                        meta['model_name'] = meta['model_name'].apply(lambda x: self.extract_value(x, tag='en'))
                        meta['product_type'] = meta['product_type'].apply(self.extract_value)
                        meta['bullet_point'] = meta['bullet_point'].apply(lambda x: self.concatenate_values(x) if isinstance(x, list) else None)
                        meta['color'] = meta['color'].apply(lambda x: self.concatenate_values(x) if isinstance(x, list) else None)
                        meta['style'] = meta['style'].apply(lambda x: self.extract_value(x, tag='en') if isinstance(x, list) else None)
                        meta['brand'] = meta['brand'].apply(lambda x: self.extract_value(x, tag='en') if isinstance(x, list) else None)
                        meta['item_keywords'] = meta['item_keywords'].apply(lambda x: self.concatenate_values(x) if isinstance(x, list) else None)
                        meta = meta[~meta['item_name_in_en'].isna()][["item_id", "product_type", "brand", "model_name", "item_name_in_en", "bullet_point", "color", "style", "main_image_id", "item_keywords"]]
                        print(f"#products with English title: {len(meta)}")
                        print(meta.head())
                        filtered_data.append(meta)
                        os.remove(file_path)
        df = pd.concat(filtered_data, ignore_index=True)  
        return df
    def extract_image_metadata_from_tar(self, tar_path, final_df):
        logger.info(f"Extracting image metadata from TAR file: {tar_path}")

        # Ensure the temporary directory is ready
        temp_dir = 'temp'
        logger.info("Print full directory path %s", os.path.abspath(temp_dir))
        os.makedirs(temp_dir, exist_ok=True)

        # Extract the TAR file
        with tarfile.open(tar_path, "r") as tar:
            logger.info(f"Contents of TAR file '{tar_path}': {[member.name for member in tar.getmembers()]}")
            
            # Find the images.csv.gz file within the metadata directory of the TAR archive
            csv_gz_file_found = False
            for member in tar.getmembers():
                if member.name.startswith('images/metadata/') and os.path.basename(member.name) == 'images.csv.gz':
                    csv_gz_file_found = True
                    # Extract the images.csv.gz file to the temporary directory
                    tar.extract(member, path=temp_dir)
                    csv_gz_path = os.path.join(temp_dir, member.name)
                    
                    # Read the images.csv.gz file into a DataFrame
                    image_meta = pd.read_csv(csv_gz_path)
                    
                    # Merge the dataframes based on the common column "image_id"
                    merged_dataset = final_df.merge(image_meta, left_on="main_image_id", right_on="image_id")
                    
                    # Remove the temporary images.csv.gz file
                    os.remove(csv_gz_path)
                    
                    return merged_dataset
                
            if not csv_gz_file_found:
                logger.error("images.csv.gz file not found in metadata directory of TAR archive.")

        # Remove the temporary directory
        self.clear_and_remove_directory(temp_dir)

        logger.error("Unable to extract image metadata from TAR file.")
        return None
    def clear_and_remove_directory(self, dir_path):
        """Removes a directory and all of its contents."""
        if os.path.exists(dir_path):  # Check if the directory exists
            shutil.rmtree(dir_path)  # Remove the directory and all its contents
            print(f"Directory {dir_path} and all its contents have been removed.")
        else:
            print(f"Directory {dir_path} does not exist.")

