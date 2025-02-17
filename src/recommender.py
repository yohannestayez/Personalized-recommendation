import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
sys.path.append('scripts')
from train_GNN import HybridGNN

class Recommender:
    def __init__(self):
        # Set device: GPU if available, otherwise CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load movie data and user IDs.
        self.movies = pd.read_csv("data/processed/movies_clean.csv")
        self.user_ids = pd.read_csv("data/processed/user_ids.csv")['userId'].tolist()
        
        # Load graph data and move to device.
        self.graph_data = torch.load("data/graph_data.pt", map_location=self.device)
        self.graph_data.x = self.graph_data.x.to(self.device)
        self.graph_data.edge_index = self.graph_data.edge_index.to(self.device)
        self.graph_data.edge_attr = self.graph_data.edge_attr.to(self.device)
        
        # Load and initialize the GNN model, then move it to device.
        self.gnn_model = HybridGNN(num_layers=3, initial_embedding=self.graph_data.x)
        self.gnn_model.load_state_dict(torch.load("models/gnn_model.pth", map_location=self.device))
        self.gnn_model.to(self.device)
        self.gnn_model.eval()  # Set to evaluation mode.
        
        # Load movie BERT embeddings, convert to a torch tensor, move to device, and normalize.
        movie_bert_emb = np.load("data/embeddings/movie_embeddings.npy")
        self.movie_bert_emb = torch.tensor(movie_bert_emb, dtype=torch.float32, device=self.device)
        self.movie_bert_emb = self.movie_bert_emb / self.movie_bert_emb.norm(dim=1, keepdim=True)
    
    def recommend(self, user_id, k=10):
        user_idx = self.user_ids.index(user_id)
        with torch.no_grad():
            # Compute embeddings for all nodes.
            embeddings = self.gnn_model(
                self.graph_data.x, 
                self.graph_data.edge_index, 
                self.graph_data.edge_attr
            )
        
        # Assume first len(user_ids) embeddings are users and the rest are movies.
        # Get the user's embedding and normalize it.
        user_emb = embeddings[user_idx]  # Shape: (embedding_dim,)
        user_emb_norm = user_emb / user_emb.norm(p=2)
        
        # Get GNN movie embeddings and normalize them.
        gnn_movie_emb = embeddings[len(self.user_ids):]
        gnn_movie_emb_norm = gnn_movie_emb / gnn_movie_emb.norm(dim=1, keepdim=True)
        
        # Compute cosine similarities on GPU.
        gnn_sim = torch.matmul(user_emb_norm, gnn_movie_emb_norm.t())
        bert_sim = torch.matmul(user_emb_norm, self.movie_bert_emb.t())
        
        # Combine the similarities with weighted coefficients.
        combined = 0.9 * gnn_sim + 0.1 * bert_sim
        
        # Get the indices of the top-K recommended movies.
        _, top_indices = torch.topk(combined, k)
        return self.movies.iloc[top_indices.cpu().numpy()][['title', 'genres','overview']]
    
       
if __name__ == "__main__":
    rec_sys = Recommender()
    print("Recommendations for user 200:")
    print(rec_sys.recommend(200))

