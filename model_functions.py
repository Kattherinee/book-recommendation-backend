import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import json


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    
		

    text = re.sub(r'[^\w\s]', '', text.lower())

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def get_prompt_embedding(prompt, model):
    cleaned_prompt = preprocess_text(prompt) 
    prompt_embedding = model.encode([cleaned_prompt])
    return prompt_embedding





def recommend_books_with_embeddings_prompt(prompt, embeddings, data, model, top_n=5, min_rating=2.5, genre_filter=None):
    try:

        prompt_embedding = model.encode([prompt])

        similarities = cosine_similarity(prompt_embedding, embeddings).flatten()

        sorted_indices = similarities.argsort()[::-1]

        min_rating = float(min_rating)

       
        if isinstance(genre_filter, str):
            genre_filter = [genre_filter.strip()] if genre_filter else []

        recommendations = []
        for idx in sorted_indices:
            book = data.iloc[idx]
            book_rating = book['rating']

            
            if isinstance(book['genres'], str):
                try:
                    book_genres = eval(book['genres'])
                except Exception as e:
                    print(f"Error parsing genres: {book['genres']}, {e}")
                    book_genres = []
            else:
                book_genres = book['genres'] or []

            
            if book_rating < min_rating:
                continue
            if genre_filter and not any(genre in book_genres for genre in genre_filter):
                continue


            recommendations.append({
                'title': book['title'],
                'author': book['author'],
                'rating': float(book_rating),
                'genres': book_genres,
                'similarity': float(similarities[idx]),
                'cover': book['coverImg']
            })

            if len(recommendations) >= top_n:
                break

        return recommendations

    except Exception as e:
        print(f"Error occurred during recommendation: {e}")
        return []







def recommend_books_with_embeddings(book_title, embeddings, data, top_n=5, min_rating=2.5, genre_filter=None):
    try:
        # Находим индекс книги
        book_index = data[data['title'] == book_title].index
        if book_index.empty:
            print(f"Книга '{book_title}' не найдена.")
            return []
        book_index = book_index[0]

        book_embedding = embeddings[book_index].reshape(1, -1)

        # Вычисляем косинусное сходство
        similarities = cosine_similarity(book_embedding, embeddings).flatten()
        sorted_indices = similarities.argsort()[::-1]

        # Преобразуем genre_filter в список (если это строка)
        if isinstance(genre_filter, str):
            genre_filter = [genre.strip() for genre in genre_filter.split(",")] if genre_filter else []

        recommendations = []
        for idx in sorted_indices:
            if idx == book_index:
                continue

            book = data.iloc[idx]
            book_rating = book['rating']

            # Обработка жанров книги
            if isinstance(book['genres'], str):
                try:
                    book_genres = eval(book['genres']) if book['genres'].startswith("[") else book['genres'].split(", ")
                except Exception as e:
                    print(f"Error parsing genres for book '{book['title']}': {e}")
                    book_genres = []
            else:
                book_genres = book['genres'] if isinstance(book['genres'], list) else []

            print(f"Processing book: {book['title']}, Genres: {book_genres}, Genre Filter: {genre_filter}")

            # Применяем фильтры
            if book_rating < min_rating:
                continue
            if genre_filter and not any(genre in book_genres for genre in genre_filter):
                continue

            recommendations.append({
                'title': book['title'],
                'author': book['author'],
                'rating': float(book_rating),
                'genres': book_genres,
                'similarity': float(similarities[idx]),
                'cover': book.get('coverImg', 'https://via.placeholder.com/150')
            })

            if len(recommendations) >= top_n:
                break

        print(f"Recommendations: {recommendations}")
        return recommendations
    except Exception as e:
        print(f"Error occurred during recommendation: {e}")
        return []

