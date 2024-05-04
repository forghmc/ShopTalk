from flask import Flask, request, jsonify
import openai
import json
from pinecone import Pinecone
import logging
import os
from flask import abort
import pandas as pd 
app = Flask(__name__)

openai.api_key= os.environ.get("OPENAI_API_KEY")
#print(open.api_key)
print("Program started")
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
    filename = f"/Users/user/Documents/MLProjects/project6/ShopTalk/artifacts/data_ingestion/data_tar_extracted/resize/{path}"
    return filename

def generate_summary(obj):
    return "hello this is the product summary"

@app.route('/llm/generate', methods=['POST'])
def generate_response():
    
    logging.info("Inside the generate_response method")
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
        response_text = response.choices[0].message.content
        response_data = json.loads(response_text)

        embedding_model = openai.embeddings.create(input=[response_text], model="text-embedding-ada-002")
        embeddings = embedding_model.data[0].embedding

        pc = Pinecone(api_key=os.environ.get("PINE_CONE_API_KEY"))
        index = pc.Index("shopping-index")

        # Query Pinecone with the generated embedding
        results = index.query(vector=embeddings, top_k=3, include_metadata=True, namespace="shopping", include_values=True)

        matches = []
        for match in results['matches']:
            matches.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            })
        images =[]
        captions = []
        sampled_data = pd.read_csv('artifacts/data_ingestion/data_tar_extracted/processed_dataset_target_data_with_captions_only.csv')
        for result in results['matches']:
            print (result)
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
        return jsonify({'images': images, 'captions': captions})
    except Exception as e:
        abort(500, description=str(e))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)