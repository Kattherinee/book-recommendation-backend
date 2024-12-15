from flask import Flask, request, jsonify
import nltk
import os

nltk.download('punkt_tab')
nltk.download('wordnet')

from flask_cors import CORS
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from model_functions import (
    recommend_books_with_embeddings_prompt,
    recommend_books_with_embeddings
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})



data_cleaned = pd.read_csv('data_cleaned.csv')
description_embeddings = np.load('description_embeddings.npy')
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def home():
    return "Welcome to the Book Recommendation API. Use /recommend/prompt or /recommend/book."


@app.route('/recommend/prompt', methods=['POST'])
def recommend_by_prompt():
    data = request.json
    prompt = data.get('prompt')
    top_n = data.get('top_n', 5)
    min_rating = data.get('min_rating', 2.5)
    genre_filter = data.get('genre_filter', None)

    print(f"Dataset rows: {data_cleaned.shape[0]}")
    print(f"Embeddings shape: {description_embeddings.shape}")

    print(f"Received data: {data}")



    print(f"Received prompt: {prompt}, top_n: {top_n}, min_rating: {min_rating}, genre_filter: {genre_filter}")
    try:
        recommendations = recommend_books_with_embeddings_prompt(
            prompt=prompt,
            embeddings=description_embeddings,
            data=data_cleaned,
            model=model,
            top_n=top_n,
            min_rating=min_rating,
            genre_filter=genre_filter
        )
        print("Recommendations generated successfully:", recommendations)
        
        return jsonify(recommendations)
    except Exception as e:
        print("Error occurred during recommendation:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/recommend/book', methods=['POST'])
def recommend_by_book():
    data = request.json
    book_title = data.get('book_title')
    top_n = data.get('top_n', 5)
    min_rating = float(data.get('min_rating', 2.5))
    genre_filter = data.get('genre_filter', None)  

    print(f"Received genres for filtering: {genre_filter}")

    recommendations = recommend_books_with_embeddings(
        book_title=book_title,
        embeddings=description_embeddings,
        data=data_cleaned,
        top_n=top_n,
        min_rating=min_rating,
        genre_filter=genre_filter  
    )
    return jsonify(recommendations)




@app.route('/genres', methods=['GET'])
def get_genres():
    try:
        genres_series = data_cleaned['genres'].apply(lambda x: eval(x) if isinstance(x, str) else [])
        all_genres = [genre for genres in genres_series for genre in genres]
        genre_counts = pd.Series(all_genres).value_counts()

        top_genres = genre_counts.head(20).index.tolist()

        return jsonify({'genres': top_genres + ['Все жанры']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
