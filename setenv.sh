export OPENAI_API_KEY=<Key insert>

export PINE_CONE_API_KEY=<Key insert>

export cohere_api_key=<Key insert>

export Dataset=artifacts/data_ingestion/data_extracted/processed_dataset_target_data_with_captions_only.csv

# MAke sure it points to the parent directory like resize or small(in case of original dataset)
export image_path=resize
echo $OPENAI_API_KEY
echo $PINE_CONE_API_KEY
echo $cohere_api_key
echo $Dataset
echo $image_path
