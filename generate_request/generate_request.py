from flask import Flask, request, jsonify
import openai
import cohere
import json
from pinecone import Pinecone
import os
from flask import abort
import pandas as pd 
app = Flask(__name__)

#openai.api_key= os.environ.get("OPENAI_API_KEY")
openai.api_key = "sk-proj-Q7C8IAr1oFunmXV1ARbkT3BlbkFJs9g1qzbXgQX19uzWSt27"
# Set your Cohere API key
cohere_api_key = 'Ca9SoTSV33ndqRYwas1Ymc8hL0ViK1QsquamXIQ3'
df = pd.read_csv('artifacts/data_ingestion/data_tar_extracted/processed_dataset_target_data_with_captions_only.csv')
documents = df.to_dict(orient='records')

# Create a key-value dictionary where key is document['item_id']
document_dict = {doc['item_id']: doc for doc in documents}

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
    filename = f"resize/{path}"
    print(filename)
    return filename
'''
def generate_summary(obj):
    return "hello this is the product summary"
'''
def generate_summary(obj):
    print(f'Genrating Summary: {obj}')
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
    print("insummary")
    user_query = f"the $Object is {json.dumps(obj.to_json())}"
    print(user_query)
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
    return response_text

@app.route('/llm/generate', methods=['POST'])
def generate_response():
    #print("generating response from backend")
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
        #print("generating response from response_text of openai", response_text)
        embedding_model = openai.embeddings.create(input=[response_text], model="text-embedding-ada-002")
        embeddings = embedding_model.data[0].embedding
        #print("generating embeddings from openai")
        pc = Pinecone(api_key=os.environ.get("PINE_CONE_API_KEY"))
        index = pc.Index("shopping-index")
     
        # Query Pinecone with the generated embedding
        results = index.query(vector=embeddings, top_k=3, include_metadata=True, namespace="shopping", include_values=True)
        #print("generating matching query embeddings with the backend embeddings from pinecone")
        matches = results['matches']
        [json.dumps(document_dict[match['id']]) for match in matches]
        match_docs = [json.dumps(document_dict[match['id']]) for match in matches]
        #print(match_docs)
        query_doc = response_text
        # Use Cohere reranking model
        co = cohere.Client(cohere_api_key)

        texts = [query_doc] + match_docs
        #print("***************************************")
        #print(texts)
        #print("***************************************")
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
        #print("Pre-loading data.csv")
        sampled_data = pd.read_csv('artifacts/data_ingestion/data_tar_extracted/processed_dataset_target_data_with_captions_only.csv')
        #print(sampled_data)
        for result in reranked_matches.results[1:4]:
            #print (result)
            item_score = result.relevance_score
            #print(item_score)
            item_text= json.loads(result.document.text)
            item_id = item_text['item_id']
            item_details = sampled_data[sampled_data['item_id'] == item_id].iloc[0]
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