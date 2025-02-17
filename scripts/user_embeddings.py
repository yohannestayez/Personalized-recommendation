import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def compute_user_embeddings():
    ratings = pd.read_csv("data/processed/ratings_clean.csv")
    movies = pd.read_csv("data/processed/movies_clean.csv")
    movie_emb = np.load("data/embeddings/movie_embeddings.npy")
    
    movie_id_to_idx = {int(float(m_id)): idx for idx, m_id in enumerate(movies['id'])}
    user_embeddings = []
    user_ids = []
    x=0
    for user_id, group in tqdm(ratings.groupby('userId'), desc="Processing users"):
        valid_indices = []
        weights = []
        
        for _, row in group.iterrows():
            movie_id=int(float(row['movieId']))
            if movie_id in movie_id_to_idx:
                valid_indices.append(movie_id_to_idx[movie_id])
                weights.append(row['rating'])
                x+=1
                
            
                
        if valid_indices:
            user_emb = np.average(movie_emb[valid_indices], axis=0, weights=weights)
            user_embeddings.append(user_emb)
            user_ids.append(user_id)
    print(f'number of appended weights: {x}')
    os.makedirs("data/embeddings", exist_ok=True)
    np.save("data/embeddings/user_embeddings.npy", np.array(user_embeddings))
    pd.Series(user_ids, name="userId").to_csv("data/processed/user_ids.csv", index=False)

if __name__ == "__main__":
    compute_user_embeddings()