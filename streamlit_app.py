import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your data
@st.cache
def load_data():
    books = pd.read_csv('C:/Users/kl/Desktop/Novel Recommender/books_cleaned.csv')
    ratings = pd.read_csv('C:/Users/kl/Desktop/Novel Recommender/ratings.csv')
    return books, ratings

books, ratings = load_data()

# Preprocess data
def preprocess_data(books, ratings):
    # Merge books and ratings data
    book_ratings = pd.merge(books, ratings, on='book_id')
    
    # Create a user-item matrix
    user_item_matrix = book_ratings.pivot_table(index='user_id', columns='title', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    
    # Compute TF-IDF matrix for book titles
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books['title'])
    
    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return user_item_matrix, cosine_sim

user_item_matrix, cosine_sim = preprocess_data(books, ratings)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = books[books['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    return books['title'].iloc[book_indices]

# Streamlit app
def main():
    st.title('Novel Recommendation System')
    
    # User input
    st.sidebar.header('User Input')
    user_id = st.sidebar.number_input('Enter User ID', min_value=1, max_value=ratings['user_id'].max(), value=1)
    
    # Get user's rated books
    user_ratings = ratings[ratings['user_id'] == user_id]
    user_rated_books = user_ratings.merge(books, on='book_id')
    
    if not user_rated_books.empty:
        st.sidebar.write('Books rated by the user:')
        st.sidebar.write(user_rated_books[['title', 'rating']])
        
        # Get recommendations based on the highest rated book
        highest_rated_book = user_rated_books.loc[user_rated_books['rating'].idxmax()]['title']
        recommendations = get_recommendations(highest_rated_book)
        
        st.write('Recommended novels based on your highest rated book:')
        st.write(recommendations)
    else:
        st.sidebar.write('No books rated by this user.')

if __name__ == '__main__':
    main()