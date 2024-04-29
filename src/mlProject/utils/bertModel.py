#import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import pandas as pd
from PIL import Image as PILImage
import numpy as np
# Initialize BERT for text
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
def preprocess_text(text):
    print(" Generating embeddings with bert preprocessor")
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    return outputs.pooler_output.detach().numpy()
 