import pandas as pd

def check_common_movie_ids():
    ratings = pd.read_csv("data/processed/ratings_clean.csv")
    movies = pd.read_csv("data/processed/movies_clean.csv", low_memory=False)
    
    # Convert both columns to a consistent integer format.
    ratings_ids = set(ratings['movieId'].apply(lambda x: int(float(x))))
    movies_ids = set(movies['id'].apply(lambda x: int(float(x))))
    
    common_ids = ratings_ids.intersection(movies_ids)
    
    print("Unique movie IDs in ratings dataset:", len(ratings_ids))
    print("Unique movie IDs in movies dataset:", len(movies_ids))
    print("Mutual movie IDs:", len(common_ids))

if __name__ == "__main__":
    check_common_movie_ids()