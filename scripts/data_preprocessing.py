# Preprocess Movies Data
import ast

def parse_genres(genre_string):
    try:
        genres = ast.literal_eval(genre_string)  # Convert string to list of dictionaries
        return [g['name'] for g in genres]  # Extract genre names
    except (ValueError, SyntaxError):
        return []