import streamlit as st
import os
import requests
st.set_option('logger.level', 'debug')
def generate_image_url(path):
    return f"/Users/user/Documents/MLProjects/project6/artifacts/data_ingestion/abo-images-small/images/resize/{path}"

def fetch_item_from_userquery(query):
    """Send a query to the Flask API and return the results."""
    url = 'http://generate_request:5000/llm/generate'  # Updated URL
    data = {'query': query}
    response = requests.post(url, json=data)
    print(response)
    if response.status_code == 200:
        return response.json()
    else:
        st.error('Failed to retrieve data from the API')
        return None

def clear_form():
    st.session_state["product_name"] = ""
    st.session_state.pop('results', None) 
def ShopTalk():
    st.title("Product Search Engine")
    query = st.text_input("Product Name:", key="product_name")

    if st.button("Search"):
        results = fetch_item_from_userquery(query)
        if results:
            images = results.get('images', [])
            captions = results.get('captions', [])
            summaries = results.get('summaries',[])
            for image_url, caption, summary in zip(images, captions, summaries):
                col1, col2 = st.columns([1, 3])
                with col1:
                    try:
                        print(image_url)
                        st.image(image_url, width=150, output_format='auto')
                    except Exception as e:
                        st.error(f"Failed to load image: {e}")

                with col2:
                    st.write(caption)
                    st.write(summary)

    clear_clicked = st.button("Clear")
    if clear_clicked:
        # Clear the session state and the input box
        st.session_state.product_name = ""
        st.experimental_rerun()

if __name__ == "__main__":
    ShopTalk()