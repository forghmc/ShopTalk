import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import pandas as pd
from PIL import Image as PILImage
import numpy as np
# Initialize BERT for text
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')