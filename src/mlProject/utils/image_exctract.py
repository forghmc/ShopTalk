from PIL import Image
import os
import pandas as pd

def resize_image(image, size=(224, 224)):
    """Resize an image to the specified size."""
    return image.resize(size)

def read_image_from_local(path):
    """Read an image file from the local filesystem."""
    try:
        print(f"Attempting to read from local path: {path}")
        image = Image.open(path)
        print("Image successfully loaded from local path")
        return image
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def save_image_to_local(image, path):
    """Save a PIL Image to the local filesystem."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
        image.save(path, format='JPEG')
        print(f"Image successfully saved to {path}")
    except IOError as e:
        print(f"An error occurred while saving the file: {e}")

def process_image(input_path, output_path):
    """Process image from local, resize and save it back to local."""
    image = read_image_from_local(input_path)
    if image is None:
        return None
    resized_image = resize_image(image)
    save_image_to_local(resized_image, output_path)
    return output_path

def process_and_store(data_row, base_input_path, base_output_path):
    input_path = os.path.join(base_input_path, data_row['path'].strip())
    output_path = os.path.join(base_output_path, data_row['path'].strip())
    
    resized_path = process_image(input_path, output_path)
    if resized_path is None:
        print(f"Skipping processing for {input_path} due to an error.")
        return None

    image = read_image_from_local(resized_path)
    if image is None:
        print(f"Failed to process image for {input_path}, skipping...")
        return None
'''
    #inputs = processor(text=[data_row['item_name_in_en_us']], images=image, return_tensors="pt", padding=True)
    #outputs = model(**inputs)
    #embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling

    # Serialize the embedding to binary using pickle
    embedding_bin = serialize_embedding(embedding)

    with engine.connect() as conn:
        ins = embeddings.insert().values(
            item_id=data_row['item_id'],
            embedding=embedding_bin,  # Store the serialized binary
            item_name=data_row['item_name_in_en_us'],
            image_path=resized_path
        )
        try:
            conn.execute(ins)
        except Exception as e:
            print(f"Failed to insert data into the database: {e}")
'''
# Data loading and processing
base_input_path = '/Users/user/Documents/MLProjects/project6/artifacts/data_ingestion/abo-images-small/images/small'
base_output_path = '/Users/user/Documents/MLProjects/project6/artifacts/data_ingestion/abo-images-small/images/resize'
dataset = pd.read_csv('/Users/user/Downloads/dataset.csv')
dataset.head(5)
dataset.apply(lambda x: process_and_store(x, base_input_path, base_output_path), axis=1)
