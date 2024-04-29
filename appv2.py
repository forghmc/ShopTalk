import streamlit as st
from PIL import Image
import requests
from io import BytesIO
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

load_dotenv()
#TO DO: Remove Hardcoded paths and index name.
pkey = os.environ.get("pkey")
pc=Pinecone(api_key=pkey)
index = pc.Index('stv1-embeddings')
sampled_data = pd.read_csv('artifacts/data_ingestion/data_tar_extracted/processed_dataset_target_data_with_captions_only.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO: Add path to config and use untar image
tarfilepath = 'artifacts/data_ingestion/abo-images-small.tar'
# Placeholder for a function that processes the query and returns image URLs and captions
def generate_image_url(path):
    # Dummy function to simulate image URL generation
 
    return f"/Users/user/Documents/MLProjects/project6/artifacts/data_ingestion/abo-images-small/images/resize/{path}"

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
    print("Fteched Product: ", caption, "Image: :", image_url)
    return images, captions

# Function to load image from URL
def load_image(url):
    response = url
    img = Image.open(BytesIO(response.content))
    return img

def app():
    st.title("Shop Talk - Product Search")
    query = st.text_input("Product Name:", "")

    if st.button("Search"):
        if query:
            print("Starting New Search")
            images, captions = search_similar_products(query)
            for image_url, caption in zip(images, captions):
                # Create a row for each product
                col1, col2 = st.columns([1, 3])
                with col1:
                    # Display image in the first column
                    st.image(image_url, width=150)  # Set width according to your layout needs
                with col2:
                    # Display caption in the second column
                    st.write(caption)
        else:
            st.write("Please enter a query to search for images.")


if __name__ == "__main__":
    app()