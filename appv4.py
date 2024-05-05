from flask import Flask, request, jsonify
import openai
import json
from pinecone import Pinecone
import os

app = Flask(__name__)

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

    # Set your OpenAI API key
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    # Send the chat completion request
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0
    )
    response_text = response.choices[0].message.content
    response_data = json.loads(response_text)

    # Your remaining code here

    return jsonify({'matches': matches})

if __name__ == '__main__':
    app.run(debug=True)


 # curl -X POST http://localhost:5000/llm/generate -H 'Content-Type: application/json' -d '{"query": "I want to buy a red t-shirt"}'