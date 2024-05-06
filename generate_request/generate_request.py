from flask import Flask, request, jsonify
import openai
import cohere
import json
from pinecone import Pinecone
import os
from flask import abort
import pandas as pd 
import logging
import sys

# Configuration for logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Directory and file paths for log storage
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Basic logging configuration with a file handler and a stream handler
logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath), # Log to a file
        logging.StreamHandler(sys.stdout) # Log to the console
    ]
)

# Logger instance with a specific name for the project
logger = logging.getLogger("ShopTalkProjectLogger")

app = Flask(__name__)

#openai.api_key= os.environ.get("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")
# Set your Cohere API key
cohere_api_key = os.environ.get('cohere_api_key')
pinecone_api_key = os.environ.get('PINE_CONE_API_KEY')

df = pd.read_csv(os.environ.get('Dataset'))
documents = df.to_dict(orient='records')

# Create a key-value dictionary where key is document['item_id']
document_dict = {doc['item_id']: doc for doc in documents}

# Define the system prompt
system_message = """
You have been assigned the task of processing user requests presented in natural language and converting them into structured data for further analysis.
Instructions:
1. Identify and extract entities (like names, organizations, products, etc.) from the user's request ($request). Store these entities in a variable called entities.
2. Identify the user's intentions behind $request. Extract these intentions and store them in a variable named intents.
3. Analyze $request to find synonyms and linguistic variations. Create a normalized version of the query that maintains the original request's meaning in a standardized form. Store this normalized query in normalized_query.
4. Create an array named normalized_queries consisting of at least five different rephrasings of $request. Each variant should represent the same intents but with different synonyms and tones.
5. Ensure the result is formatted as a JSON string only, containing no extra text or formatting.
"""
def generate_image_url(path):
    filename = f"{os.environ.get('image_path')}/{path}"
    return filename
'''
def generate_summary(obj):
    return "hello this is the product summary"
'''
def generate_summary(obj):
    # Define the system prompt and user query
    system_message = """
    You are a expert at generating a concise summary given a $Object json:
    Instructions:

    1.The object has some keys which include ["item_id", "product_type", "brand", "model_name", "item_name_in_en_us", "bullet_point", "color", "style", "item_keywords", path]
    2. Your task is to consider the above and present the summary in the following way
    characteristics: summarise in natural language the color and style
    3. Summarise the bullet_point in less than or equal to 3 sentences
    4. The output should string containing the concise summary. You can also add a stylish statement for marketing the product. Keep the statement apt to the product.
    """
    logger.info("Generating summary from the engine")
    user_query = f"the $Object is {json.dumps(obj.to_json())}"
    
    logger.info("Gemerated Summary")

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
    logger.info("Print response from openAI %s", response_text)
    return response_text

@app.route('/llm/generate', methods=['POST'])
def generate_response():
    print("generatimg 1")
    try: 
        user_query = request.get_json().get('query')

        
        # Send the chat completion request
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0
        )
        print("generatimg 2")
        response_text = response.choices[0].message.content
        response_data = json.loads(response_text)
        embedding_model = openai.embeddings.create(input=[response_text], model="text-embedding-ada-002")
        embeddings = embedding_model.data[0].embedding
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index("shopping-index")
        print("generatimg 3")
        # Query Pinecone with the generated embedding
        results = index.query(vector=embeddings, top_k=3, include_metadata=True, namespace="shopping", include_values=True)
        matches = results['matches']
        [json.dumps(document_dict[match['id']]) for match in matches]
        match_docs = [json.dumps(document_dict[match['id']]) for match in matches]
        query_doc = response_text
        # Use Cohere reranking model
        co = cohere.Client(cohere_api_key)
        print("generatimg 4")
        texts = [query_doc] + match_docs
        reranked_matches = co.rerank(
            query=query_doc,
            documents=texts,
            top_n=4,
            model="rerank-english-v2.0",
            return_documents=True
        )
        images =[]
        captions = []
        summaries = []
        print("generatimg 5")
        for result in reranked_matches.results[1:4]:
            print("generatimg 6")
            item_score = result.relevance_score
            item_text= json.loads(result.document.text)
            item_id = item_text['item_id']
            item_details = df[df['item_id'] == item_id].iloc[0]
            item_name = item_details['item_name_in_en']
            brand_name = item_details['brand']
            product_type = item_details['product_type']
            item_image_path = item_details['path']
            image_url = generate_image_url(item_image_path)
            summary = generate_summary(item_details)
            summaries.append(summary)
            images.append(image_url)
            captions.append(f"Product: {item_name}, Brand: {brand_name}, Type: {product_type}, Relevance Score: {item_score}")
            print (captions)    
        return jsonify({'images': images, 'captions': captions, 'summaries': summaries})
    except Exception as e:
        abort(500, description=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)