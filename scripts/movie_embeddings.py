import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

def generate_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    model = BertModel.from_pretrained("bert-large-uncased").to(device)
    model.eval()
    
    movies = pd.read_csv("data/processed/movies_clean.csv")
    texts = movies["bert_text"].tolist()
    
    embeddings = []
    for i in tqdm(range(0, len(texts)), desc="Generating BERT embeddings"):
        batch = texts[i:i+32]
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:,0,:].cpu().numpy())
    
    movie_embeddings = np.concatenate(embeddings)
    os.makedirs("data/embeddings", exist_ok=True)
    np.save("data/embeddings/movie_embeddings.npy", movie_embeddings)

if __name__ == "__main__":
    generate_embeddings()