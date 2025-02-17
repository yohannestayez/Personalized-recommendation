import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split
import numpy as np
import os

class LightGNN(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        
    def forward(self, x, edge_index, edge_weight):
        # Compute degrees based on the full graph (x contains all node features)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        # Clamp degree to avoid division by zero and NaN values
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight.squeeze()
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class HybridGNN(nn.Module):
    def __init__(self, num_layers, initial_embedding):
        super().__init__()
        # Global learnable node embeddings.
        self.embedding = nn.Parameter(initial_embedding)
        self.layers = nn.ModuleList([LightGNN() for _ in range(num_layers)])
        
    def forward(self, x, edge_index, edge_weight):
        embeddings = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            embeddings.append(x)
            print(f"[Forward] After layer {i+1}, embedding shape: {x.shape}")
        final_embedding = torch.mean(torch.stack(embeddings), dim=0)
        print("[Forward] Final embedding shape:", final_embedding.shape)
        return final_embedding

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading data...")
    data = torch.load("data/graph_data.pt")
    print("Graph data loaded successfully.")
    print("Edge index shape:", data.edge_index.shape)
    print("Edge index sample:", data.edge_index[:, :10])
    
    # Move node features and edge information to the device.
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_attr = data.edge_attr.to(device)
    
    # Convert edges and edge weights to numpy arrays for train-test splitting.
    edges = data.edge_index.t().cpu().numpy()  # shape: (num_edges, 2)
    edge_weights = data.edge_attr.cpu().numpy()
    
    # Train-test split on edges.
    edges_train, edges_test, weights_train, weights_test = train_test_split(
        edges, edge_weights, test_size=0.2, random_state=42
    )
    print(f"Train edges: {edges_train.shape[0]}, Test edges: {edges_test.shape[0]}")
    
    # Build a training Data object using only training edges.
    train_edge_index = torch.tensor(edges_train, dtype=torch.long, device=device).t()
    train_edge_attr = torch.tensor(weights_train, dtype=torch.float, device=device)
    train_data = Data(x=data.x, edge_index=train_edge_index, edge_attr=train_edge_attr)
    
    # Initialize model with global embeddings.
    model = HybridGNN(num_layers=3, initial_embedding=data.x)
    model = model.to(device)
    print("Model initialized with node embeddings of shape:", data.x.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Assumption: first half of nodes are users and the rest are movies.
    num_users = data.x.size(0) // 2
    
    # Training loop using the full graph.
    for epoch in range(30):
        print(f"\n=== Epoch {epoch+1} ===")
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on the full training graph.
        x_full = model.embedding  # global node features
        out = model(x_full, train_data.edge_index, train_data.edge_attr)
        
        # BPR Loss Computation on the full graph.
        global_source = train_data.edge_index[0]
        global_target = train_data.edge_index[1]
        
        # Identify valid user->movie edges (users: index < num_users, movies: index >= num_users).
        valid_mask = (global_source < num_users) & (global_target >= num_users)
        if valid_mask.sum() == 0:
            print("[Train] No valid user->movie edges in this epoch.")
            continue
        
        # Use valid edges directly since indices already correspond to global indices.
        user_emb = out[global_source[valid_mask]]
        pos_emb = out[global_target[valid_mask]]
        
        # Negative sampling: sample negative movie nodes from nodes that are movies.
        movie_indices = torch.arange(num_users, out.size(0), device=device)
        neg_sample_indices = movie_indices[torch.randint(0, movie_indices.size(0), (valid_mask.sum(),), device=device)]
        neg_emb = out[neg_sample_indices]
        
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        print("[Train] Loss:", loss.item())
        
        loss.backward()
        optimizer.step()
        
        # ----- Validation using full-graph inference -----
        if epoch % 5 == 0:
            print("[Validation] Starting validation...")
            model.eval()
            with torch.no_grad():
                # Compute full graph embeddings.
                full_embeddings = model(model.embedding, data.edge_index, data.edge_attr)
                
                # Calculate Hit@10 using cosine similarity threshold as an example.
                hit = 0
                for idx, (u, m) in enumerate(edges_test):
                    user_emb_val = full_embeddings[u]
                    movie_emb_val = full_embeddings[m]
                    sim = F.cosine_similarity(user_emb_val.unsqueeze(0), movie_emb_val.unsqueeze(0))
                    if sim > 0.5:
                        hit += 1
                    if idx < 5:
                        print(f"[Validation] Sample {idx}: user {u}, movie {m}, similarity: {sim.item():.4f}")
                hit_rate = hit / len(edges_test)
                print(f"[Epoch {epoch+1}] Validation Hit@10: {hit_rate:.4f}")
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/gnn_model.pth")
    print("Model saved at models/gnn_model.pth")

if __name__ == "__main__":
    train()
