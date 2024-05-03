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
from mlProject.utils.common import fetch_item_from_userquery

load_dotenv()
pkey = os.environ.get("PINE_CONE_API_KEY")
#pc = Pinecone(api_key=pkey)
#index = pc.Index('stv1-embeddings')

pc = Pinecone(api_key=pkey)
index = pc.Index("shopping-index")

#sampled_data = pd.read_csv('artifacts/data_ingestion/data_tar_extracted/processed_dataset_target_data_with_captions_only.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_image_url(path):
    return f"/Users/user/Documents/MLProjects/project6/artifacts/data_ingestion/abo-images-small/images/resize/{path}"

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
            st.session_state.results = fetch_item_from_userquery(query)
             #query_embedding = fetch_item_from_userquery(query)
 
        if 'results' in st.session_state and st.session_state.results:
            for image_url, caption in zip(st.session_state.results[0], st.session_state.results[1]):
                col1, col2 = st.columns([1, 3])
                with col1:
                    try:
                        if image_url:
                            st.image(image_url, width=150)
                    except Exception as e:  # Catching a general exception to handle any error while loading the image
                            st.error(f"Failed to load image: {e}") 
 
                with col2:
                    #formatted_caption = caption.replace(", ", "\n")  # Replace commas with newline characters
                    st.write(caption)
                    
if __name__ == "__main__":
    ShopTalk()
