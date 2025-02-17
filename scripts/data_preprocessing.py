import os
import ast
import pandas as pd
import re
from bs4 import BeautifulSoup

# Preprocess Movies Data
import ast

def parse_genres(genre_string):
    try:
        genres = ast.literal_eval(genre_string)  # Convert string to list of dictionaries
        return [g['name'] for g in genres]  # Extract genre names
    except (ValueError, SyntaxError):
        return []

def parse_json_field(field):
    try:
        return [item['name'] for item in ast.literal_eval(field)]
    except:
        return []

def clean_text(text):
    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def process_data(movies, ratings):
    # Movies
    # movies = pd.read_csv("data/raw/movies_metadata.csv", low_memory=False)
    json_cols = ['production_companies', 'production_countries', 'spoken_languages']
    for col in json_cols:
        movies[col] = movies[col].apply(parse_json_field)
    movies['release_year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    movies['overview'] = movies['overview'].apply(clean_text).fillna('')
    movies['tagline'] = movies['tagline'].apply(clean_text).fillna('')
    movies['bert_text'] = movies.apply(
        lambda x: f"[TITLE]{x['title']}[YEAR]{x['release_year']}"
        f"[GENRES]{','.join(x['genres'])}[OVERVIEW]{x['overview']}"
        f"[TAGLINE]{x['tagline']}", axis=1
    )

    movies['id'] = movies['id'].astype(str)
    ratings['movieId'] = ratings['movieId'].astype(str)
    
   
    
    # Ratings
    # ratings = pd.read_csv("data/raw/ratings_small.csv")
    ratings = ratings.dropna().drop_duplicates()
    ratings['rating'] = ratings['rating'] / 5.0
    

from datetime import datetime

def check_movie_ids_consistency(movies):
    date_ids = []
    non_date_ids = []
    
    for mid in movies['id']:
        mid_str = str(mid)
        try:
            # If this works, then mid is a date in the expected format.
            datetime.strptime(mid_str, "%Y-%m-%d")
            date_ids.append(mid_str)
        except ValueError:
            non_date_ids.append(mid_str)
    
    if date_ids:
        print("Found date-like IDs in movies")
        for d in date_ids:
            print(d)
        # Drop movies with date-like IDs
        movies.drop(movies[movies['id'].isin(date_ids)].index, inplace=True)
        print(f"Dropped {len(date_ids)} movies with date-like IDs.")
    else:
        print("No date-like movie IDs found.")



# if __name__ == "__main__":
#     os.makedirs("data/processed", exist_ok=True)
#     process_data()