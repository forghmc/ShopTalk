from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from pathlib import Path
from mlProject.entity.config_entity import DataIngestionConfig
from mlProject.entity.config_entity import DataValidationConfig
from mlProject.entity.config_entity import DataTransformationConfig
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.entity.config_entity import ModelEvaluationConfig

class ConfigurationManager:
    # Base directory where the script is located
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

    # Configuration, Parameters, and Schema file paths
    CONFIG_FILE_PATH = BASE_DIR / "config/config.yaml"
    PARAMS_FILE_PATH = BASE_DIR / "params.yaml"
    SCHEMA_FILE_PATH = BASE_DIR / "schema.yaml"
    def __init__(
        self,
        config_filepath = Path(CONFIG_FILE_PATH),
        params_filepath = Path(PARAMS_FILE_PATH),
        schema_filepath = Path(SCHEMA_FILE_PATH)):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_data_url =config.source_data_url,
            source_image_url =config.source_image_url,
            local_data_file =config.local_data_file,
            local_image_file =config.local_image_file,
            target_data_file =config.target_data_file,
            target_data_image_file =config.target_data_image_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            input_root= config.input_root,
            root_dir=config.root_dir,
            input_file = config.input_file,
            output_file = config.output_file,
            STATUS_FILE=config.STATUS_FILE,
            untar_data_dir = config.untar_data_dir,
            all_schema=schema,
        )
    
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_dir=config.data_dir,
            base_dir=config.base_dir,
            input_file = config.input_file,
            output_file = config.output_file,
            metadata_dir=config.metadata_dir,
            ingest_dir=config.ingest_dir
        )

        return data_transformation_config
    
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.ElasticNet
        schema =  self.schema.TARGET_COLUMN
    

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
        config = self.config.model,
        root_dir=config.root_dir,
        input_file = config.input_file,
        output_file_captions = config.output_file,
        final_output_file=config.metadata_dir,
        ingest_dir=config.ingest_dir
        )
        
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.ElasticNet
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            mlflow_uri="https://dagshub.com/forghmc/ShopTalk.mlflow",
        )

        return model_evaluation_config
