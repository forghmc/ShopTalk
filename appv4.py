from flask import Flask, request, jsonify
import openai
import json
from pinecone import Pinecone

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = ''

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

@app.route('/llm/generate', methods=['POST'])
def generate_response():
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

    pc = Pinecone(api_key="")
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

        captions.append(f"Product: {item_name}, Brand: {brand_name}, Type: {product_type}")
        print (captions)    
    return jsonify({'matches': matches})

if __name__ == '_main_':
    app.run(debug=True)