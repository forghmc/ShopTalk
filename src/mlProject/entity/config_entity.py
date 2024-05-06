from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_data_url: str
    source_image_url: str 
    local_data_file: str 
    local_image_file: str 
    target_data_file : str 
    target_data_image_file: Path 
    unzip_dir: Path 

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    input_root:Path
    input_file:str
    output_file:str
    STATUS_FILE: str
    untar_data_dir: Path
    all_schema: dict



@dataclass(frozen=True)
class DataTransformationConfig:
    data_dir: Path
    base_dir: Path
    input_file: Path
    output_file: Path
    ingest_dir: Path
    root_dir: Path
    metadata_dir : Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    input_file: Path
    output_file_captions: Path
    final_output_file: str
    ingest_dir: float
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str