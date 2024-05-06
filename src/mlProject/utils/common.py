import os
from box.exceptions import BoxValueError
import yaml
from mlProject import logger
import json
import joblib
import boto3
import pandas as pd
from io import StringIO
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from PIL import Image
import requests
import torch
import io
#Enable it for runninge the models
#from transformers import BlipProcessor, BlipForConditionalGeneration
from io import BytesIO
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
#import openai
from pinecone import Pinecone

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

'''
    Note this code needs first to generate image caption(Using blip) and
    that are generated during model training and it is compute intesnsive.
    So filtering the dataset based on the dataset that will have caption generated.
    This function will be called from model trainer.
    It also generate the combined column which has text joined from one column as
    seprated by space. 
    '''
def filter_caption_generate_combined_column(dataset):
    '''
    Filters rows in a DataFrame based on the 'caption' column to exclude any rows where
    the caption is missing or empty.
    '''
    filtered_df = dataset[dataset['caption'].notna() & dataset['caption'].astype(str).str.strip() != '']
    columns_to_combine = filtered_df.columns.drop('path')

    # Safely combine the data from the selected columns into a new column with space-separated values
    filtered_df['combined'] = filtered_df[columns_to_combine].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return filtered_df
        

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"
@ensure_annotations
def read_image_from_tar(tarfile, path):
    member = tarfile.getmember(path)
    if member is None:
        print(f"File {path} not found in tar.")
        return None
    image_bytes = tarfile.extractfile(member).read()
    return Image.open(io.BytesIO(image_bytes))
@ensure_annotations
def load_product_image(img_url):
    try:
        image = img_url
        if image is None:
            return None
        image = image.convert("RGB")
        
        transform = transforms.Compose([
        transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        return image
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")
        return None
def generate_image_caption(dataset):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for index, row in dataset.iterrows():
        processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        image = load_product_image('artifacts/data_transformation/resize/' + row['path'], device)
        if image is None:
            continue
        caption =''
        with torch.no_grad():
            caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
            dataset.at[index, 'caption'] = caption 
    return dataset
    


def read_csv_from_s3(key):

    # Setup Boto3 to access S3
    s3_client = boto3.client('s3')
    bucket_name = 'shop-talk-data'

    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
    
    if status == 200:
        df = pd.read_csv(response['Body'])
        return df
    else:
        raise Exception('Failed to fetch csv file from S3')

def write_csv_to_s3(df, key):

    # Setup Boto3 to access S3
    s3_client = boto3.client('s3')
    bucket_name = 'shop-talk-data'

    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    response = s3_client.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())
    status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
    
    if status == 200:
        print('File successfully saved to S3')
    else:
        raise Exception('Failed to save csv file to S3')
    
def generate_image_url(path):
    filename = f"/artifacts/data_ingestion/abo-images-small/images/small/{path}"
    return filename

def fetch_item_from_userquery(user_query):
    # Set your OpenAI API key
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    pkey = os.environ.get("PINE_CONE_API_KEY")
    # Define the system prompt and user query
    system_message = """
    You have been assigned the task of processing user requests presented in natural language and converting them into structured data for further analysis.
    Instructions:
    1. Identify and extract entities (like names, organizations, products, etc.) from the user's request ($request). Store these entities in a variable called entities.
    2. Identify the user's intentions behind $request. Extract these intentions and store them in a variable named intents.
    3. Analyze $request to find synonyms and linguistic variations. Create a normalized version of the query that maintains the original request's meaning in a standardized form. Store this normalized query in normalized_query.
    4. Create an array named normalized_queries consisting of at least five different rephrasings of $request. Each variant should represent the same intents but with different synonyms and tones.
    5. Ensure the result is formatted as a JSON string only, containing no extra text or formatting.
    """
    if not user_query:
        user_query = "I want to buy a red t-shirt"


    # Send the chat completion request
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0
    )
    response_text = response.choices[0].message.content
    # Print the response
    print(response_text)
    print(json.loads(response_text))

    embedding_model = openai.embeddings.create(input=[response_text], model="text-embedding-ada-002")
    embeddings = embedding_model.data[0].embedding

    print(embeddings)

    pc = Pinecone(api_key=pkey)

    index = pc.Index("shopping-index")
    # Query Pinecone with the generated embedding
    results = index.query(vector=embeddings, top_k=3, include_metadata=True, namespace="shopping", include_values=True)
    print(results)
    # Display the results
    #for match in results['matches']:
    #    print(f"Score: {match['score']}, {match['id']}, Metadata: {match['metadata']}")
    images = []
    captions = []
    sampled_data = pd.read_csv('artifacts/data_ingestion/data_tar_extracted/processed_dataset_target_data_with_captions_only.csv')
    for result in results['matches']:
        item_id = result['id']
        item_details = sampled_data[sampled_data['item_id'] == item_id].iloc[0]
        item_name = item_details['item_name_in_en']
        brand_name = item_details['brand']
        product_type = item_details['product_type']
        item_image_path = item_details['path']
        image_url = generate_image_url(item_image_path)
        
        images.append(image_url)
        captions.append(f"Product: {item_name}, Brand: {brand_name}, Type: {product_type}")
        print (captions)    
    return images, captions