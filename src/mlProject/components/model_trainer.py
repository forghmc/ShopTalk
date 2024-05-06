import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
import joblib
from mlProject.utils.common import generate_image_caption
from mlProject.utils.common import filter_caption_generate_combined_column
from mlProject.entity.config_entity import ModelTrainerConfig
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
import json
from pinecone import Pinecone


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        #self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # Initialize OpenAI client
        #self.pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])  # Initialize Pinecone client
        
    def get_embedding(self,text,embeddings):
        try:
            # Generating embeddings using the specified engine
            response = embeddings.get_text_embedding(text)
            print(response)
            return response  # Extracting the embedding from the response
        except Exception as e:
            print(f"An error occurred: {str(e)}")  # Printing out the error if something goes wrong
            return None

    def initiate_model_training(self):
        try:
            # Ensure the root directory exists
            if not os.path.exists(self.config.root_dir):
                os.makedirs(self.config.root_dir)

            # Load file to dataframe
            inputfile = os.path.join(self.config.ingest_dir, os.path.basename(self.config.input_file))
            logger.info("Loading transformed database %s", inputfile)
            df_input = pd.read_csv(inputfile)

            # Generate captions (assuming generate_captions is defined elsewhere and works with df_input)
            df_captions = generate_image_caption(df_input)

            # Save the captions to a file
            outputfile_captions = os.path.join(self.config.root_dir, os.path.basename(self.config.output_file_captions))
            df_captions.to_csv(outputfile_captions, index=False)
            logger.info("Saved captions to %s", outputfile_captions)
            # Get the refined dataset that filters the dataset for the records that have captions only.
            df_refined = filter_caption_generate_combined_column(df_captions)
            documents = df_refined.to_dict(orient='records')
            #prepare the cvector store
            index_name = 'shopping-index'
            pc = Pinecone(api_key=self.pinecone_client)

            #pincone.create_index(index_name, dimension=1536)
            index = pc.Index(index_name)
            # Set up the language model and embeddings
            llm = OpenAI(model='gpt-3.5-turbo',temperature=0.0)
            embeddings = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
            for document in documents:
                full_text = json.dumps(document)  # Or other text preprocessing based on your need
                embedding = self.get_embedding(full_text,embeddings)
                logger.info("Printing embdding type %s",type(embedding))
                if embedding is not None:
                    if embedding is None or not isinstance(embedding, list):
                        print("Failed to generate a valid embedding.")
                        continue  # Skip this document or handle the error as needed
                    v = {"id":document['item_id'], "values": embedding, "metadata":{"brand":document["brand"]} }
                    # v = Vector(id=document['item_id'], values=embedding)
                    print(v)
                    # vecs.append(v)
                    index.upsert(vectors=[v], namespace="shopping")

            logger.info(" Vectors created and persisted in Picone db")
            # Extracting the embedding from the response
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")  # Logging the error
            return None


    
 

    