import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree
import numpy as np
import os

class LightGNN(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        
    def forward(self, x, edge_index, edge_weight):
        # Compute degrees based on the mini-batch subgraph
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight.squeeze()
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class HybridGNN(nn.Module):
    def __init__(self, num_layers, initial_embedding):
        super().__init__()
        # Register node embeddings as a learnable parameter.
        self.embedding = nn.Parameter(initial_embedding)
        self.layers = nn.ModuleList([LightGNN() for _ in range(num_layers)])
        
    def forward(self, edge_index, edge_weight):
        # Start from the global node embeddings.
        x = self.embedding
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
    
    # Move node features to the device
    data.x = data.x.to(device)
    
    # Convert edges and weights to numpy for batching
    edges = data.edge_index.t().cpu().numpy()  # Shape: (num_edges, 2)
    edge_weights = data.edge_attr.cpu().numpy()
    
    # Train-test split on edges
    edges_train, edges_test, weights_train, weights_test = train_test_split(
        edges, edge_weights, test_size=0.2, random_state=42
    )
    print(f"Train edges: {edges_train.shape[0]}, Test edges: {edges_test.shape[0]}")
    
    # Initialize model with trainable node embeddings and move to device
    model = HybridGNN(num_layers=3, initial_embedding=data.x)
    model = model.to(device)
    print("Model initialized with node embeddings of shape:", data.x.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = 32  # Set your desired mini-batch size here
    
    # Training loop (mini-batch over training edges)
    for epoch in range(50):
        print(f"\n=== Epoch {epoch+1} ===")
        model.train()
        # Shuffle training edges each epoch
        perm = np.random.permutation(len(edges_train))
        epoch_loss = 0.0
        num_batches = 0
        
        # Process training edges in mini-batches
        for i in range(0, len(edges_train), batch_size):
            batch_indices = perm[i: i+batch_size]
            batch_edges = edges_train[batch_indices]
            batch_weights = weights_train[batch_indices]
            
            optimizer.zero_grad()
            
            # Create mini-batch tensors and move to device
            edge_index_batch = torch.tensor(batch_edges, dtype=torch.long, device=device).t()
            edge_weight_batch = torch.tensor(batch_weights, dtype=torch.float, device=device)
            print("[Train] Mini-batch edge_index shape:", edge_index_batch.shape)
            print("[Train] Mini-batch edge_weight shape:", edge_weight_batch.shape)
            
            # Forward pass on the mini-batch subgraph
            batch_embeddings = model(edge_index_batch, edge_weight_batch)
            print("[Train] Computed embeddings shape:", batch_embeddings.shape)
            
            # BPR Loss computation for the mini-batch
            users = batch_edges[:, 0]
            # Adjust positive item indices based on the assumed movie offset
            pos_items = batch_edges[:, 1] - (data.x.size(0) // 2)
            neg_items = np.random.choice(data.x.size(0) // 2, size=len(users), replace=True)
            
            users_tensor = torch.tensor(users, dtype=torch.long, device=device)
            pos_tensor = torch.tensor(pos_items, dtype=torch.long, device=device)
            neg_tensor = torch.tensor(neg_items, dtype=torch.long, device=device)
            
            user_emb = batch_embeddings[users_tensor]
            pos_emb = batch_embeddings[pos_tensor + (data.x.size(0) // 2)]
            neg_emb = batch_embeddings[neg_tensor + (data.x.size(0) // 2)]
            
            print("[Train] User embeddings shape:", user_emb.shape)
            print("[Train] Positive embeddings shape:", pos_emb.shape)
            print("[Train] Negative embeddings shape:", neg_emb.shape)
            
            pos_scores = (user_emb * pos_emb).sum(dim=1)
            neg_scores = (user_emb * neg_emb).sum(dim=1)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            print("[Train] Mini-batch Loss:", loss.item())
            
            loss.backward()
            optimizer.step()
            print("[Train] Optimizer step completed for mini-batch.")
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"[Epoch {epoch+1}] Average Loss: {avg_epoch_loss:.4f}")
        
        # Validation every 5 epochs (full-batch validation for simplicity)
        if epoch % 5 == 0:
            print("[Validation] Starting validation...")
            model.eval()
            with torch.no_grad():
                edge_index_test = torch.tensor(edges_test, dtype=torch.long, device=device).t()
                edge_weight_test = torch.tensor(weights_test, dtype=torch.float, device=device)
                print("[Validation] edge_index_test shape:", edge_index_test.shape)
                test_embeddings = model(edge_index_test, edge_weight_test)
                print("[Validation] Test embeddings computed. Shape:", test_embeddings.shape)
                
                # Calculate Hit@10 using cosine similarity threshold
                hit = 0
                for idx, (u, m) in enumerate(edges_test):
                    user_emb_val = test_embeddings[u]
                    movie_emb_val = test_embeddings[m]
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
