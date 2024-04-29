import streamlit as st
from PIL import Image
from io import BytesIO
import os
import torch
import pandas as pd
import numpy as np
from mlProject.utils.bertModel import preprocess_text
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv()
pkey = os.environ.get("pkey")
pc = Pinecone(api_key=pkey)
index = pc.Index('stv1-embeddings')
sampled_data = pd.read_csv('artifacts/data_ingestion/data_tar_extracted/processed_dataset_target_data_with_captions_only.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_image_url(path):
    return f"/Users/user/Documents/MLProjects/project6/artifacts/data_ingestion/abo-images-small/images/resize/{path}"

def search_similar_products(query_text):
    query_embedding = preprocess_text(query_text).flatten()
    results = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)
    images = []
    captions = []
    
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
    
    return images, captions
def clear_form():
    st.session_state["product_name"] = ""
    st.session_state.pop('results', None) 
def ShopTalk():
    st.title("Product Search Engine")
    with st.form("Shop Talk"):
    
    
    
    # Use the session state to reset the input field more effectively
        if 'reset' not in st.session_state:
            st.session_state.reset = False

        if st.session_state.reset:
            query = st.text_input("Product Name:", value="", key="product_name")
            st.session_state.reset = False
        else:
            query = st.text_input("Product Name:", value="", key="product_name")

        search_clicked = st.form_submit_button(label="Search")
        clear_clicked = st.form_submit_button(label="Clear", on_click=clear_form)

        if search_clicked and query:
            st.session_state.results = search_similar_products(query)
 
        if 'results' in st.session_state and st.session_state.results:
            for image_url, caption in zip(st.session_state.results[0], st.session_state.results[1]):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(image_url, width=150)
                with col2:
                    st.write(caption)
if __name__ == "__main__":
    ShopTalk()
