import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
import json
from pinecone import Pinecone


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # Initialize OpenAI client
        self.pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])  # Initialize Pinecone client
        
    def get_embedding(self, text):
        try:
            embeddings = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
            response = embeddings.get_text_embedding(text)
            return response  # Extracting the embedding from the response
        except Exception as e:
            print(f"An error occurred: {str(e)}")  # Printing out the error if something goes wrong
            return None

    
    def train(self):
        train_data = pd.read_json(self.config.train_data_path,lines=True)
        test_data = pd.read_json(self.config.test_data_path, lines=True)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column],axis=1)
        
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]
        
        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        #lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

    