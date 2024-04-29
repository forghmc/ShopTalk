import gradio as gr
import os
import torch
import pandas as pd
import numpy as np
from mlProject.utils.bertModel import preprocess_text
from mlProject.utils.common import load_product_image
from PIL import Image
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import io
import base64
load_dotenv()
pkey = os.environ.get("pkey")
pc=Pinecone(api_key=pkey)
index = pc.Index('stv1-embeddings')
sampled_data = pd.read_csv('artifacts/data_ingestion/data_tar_extracted/processed_dataset_target_data_with_captions_only.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO: Add path to config and use untar image
tarfilepath = 'artifacts/data_ingestion/abo-images-small.tar'
# Dummy preprocess_text and generate_image_url functions
'''
def preprocess_text(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.pooler_output.detach().numpy()
'''
def generate_image_url(path):
    # Specify the directory path to where your images are stored
    base_path = "/Users/user/Documents/MLProjects/project6/artifacts/data_ingestion/abo-images-small/images/resize/"
    full_path = os.path.join(base_path, path)

    try:
        with Image.open(full_path) as img:
            # Resize the image to 224x224
            img = img.resize((224, 224), Image.Resampling.LANCZOS)

            # Save the resized image to a byte buffer
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            # Encode the image to base64 and format it as a data URL
            base64_img = base64.b64encode(img_byte_arr).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_img}"
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return None 

def search_similar_products(query_text):
    query_embedding = preprocess_text(query_text).flatten()
    results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)
    images = []
    captions = []
    
    for result in results['matches']:
        item_id = result['id']
        metadata = result.get('metadata', {})
        item_details = sampled_data[sampled_data['item_id'] == item_id].iloc[0]
        
        item_name = item_details['item_name_in_en']
        brand_name = item_details['brand']
        product_type = item_details['product_type']
        item_image_path = item_details['path']
        image_url = generate_image_url(item_image_path)
        
        images.append(image_url)
        caption = f"Product: {item_name}, Brand: {brand_name}, Type: {product_type}"
        captions.append(caption)
    
    return images, captions


def clear_all(chatbot, image_display):
    chatbot.clear()
    image_display.clear()
    return [], []
def setup_ui(query):
    try:
        images, captions = search_similar_products(query)
        gallery_data = [(image, caption) for image, caption in zip(images, captions)]
        return gallery_data
    except Exception as e:
        print(f"Error in setup_ui: {e}")
        return []  # Return an empty list in case of an error
css = """
#gallery {font-size: 24px !important}
"""
with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Enter product query:")
    submit_button = gr.Button("Submit")
    gallery = gr.Gallery(label="Product Images and Descriptions",show_label= False,
    elem_id="gallery", columns=[8], rows=[1], object_fit='scale-down', height=50)

    submit_button.click(fn=setup_ui, inputs=[input_text], outputs=[gallery])

demo.launch(height= 60, width ="50%")
'''
def setup_ui(query):
    results = search_similar_products(query)
    output_image = []
    output_data =[]
    # Check results and prepare data for output
    for result in results:
        # Append a tuple of (image_url, description) for each result
        output_image.append(result["image"])
        output_data.append(result["description"])
        print("Output Data ", output_data, output_image)
    return output_image, output_data '''




'''
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# ShopTalk Product Search")
    with gr.Row():
        input_text = gr.Textbox(label="Enter Product Name:")
    with gr.Row():
        output_text = gr.Textbox(label="Results")
    with gr.Row():
        output_image = gr.Image(label="Generated Image")
        submit_button = gr.Button("Submit")

    submit_button.click(
        fn=search_similar_products,
        inputs=input_text,
        outputs=[output_text, output_image]
    )
    clear_button = gr.Button("Clear")
    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[output_text, output_image]
    )

demo.launch()
'''
