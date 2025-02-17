import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

def build_interaction_graph():
    # Load data
    users = pd.read_csv("data/processed/user_ids.csv")
    movies = pd.read_csv("data/processed/movies_clean.csv")
    ratings = pd.read_csv("data/processed/ratings_clean.csv")
    
    # Load embeddings
    user_emb = np.load("data/embeddings/user_embeddings.npy")
    movie_emb = np.load("data/embeddings/movie_embeddings.npy")
    
    # Create mappings
    user_id_to_idx = {uid: idx for idx, uid in enumerate(users['userId'])}
    movie_id_to_idx = {mid: idx+len(user_id_to_idx) for idx, mid in enumerate(movies['id'])}
    
    # Build edges with weights
    edge_indices = []
    edge_weights = []
    for _, row in ratings.iterrows():
        u_idx = user_id_to_idx.get(row['userId'], -1)
        m_idx = movie_id_to_idx.get(row['movieId'], -1)
        if u_idx != -1 and m_idx != -1:
            edge_indices.append([u_idx, m_idx])
            edge_weights.append(row['rating'])
    
    # Create PyG data object
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    x = torch.cat([
        torch.tensor(user_emb, dtype=torch.float),
        torch.tensor(movie_emb, dtype=torch.float)
    ], dim=0)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    torch.save(data, "data/graph_data.pt")

if __name__ == "__main__":
    build_interaction_graph()