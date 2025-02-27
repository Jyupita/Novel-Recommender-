import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set wide layout for the app
st.set_page_config(layout="wide")

# Custom CSS for full-width expansion
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 100%;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    # Streamlit UI Header
    st.title('📚 Novel Recommendation System')
    st.markdown("""
    Welcome to the **Novel Recommendation System**! This app recommends novels based on your reading preferences. 
    Enter your User ID in the sidebar to get started.
    """)

    # Sidebar - About the App
    st.sidebar.header('About the App')
    st.sidebar.markdown("""
    This app uses a **content-based recommendation system** to suggest novels similar to the ones you've rated highly. 
    It analyzes book titles and user ratings to provide personalized recommendations.
    """)

    # Sidebar - Technical Details
    st.sidebar.header('Technical Details')
    st.sidebar.markdown("""
    - **Data**: The app uses two datasets: `books_cleaned.csv` (book details) and `ratings.csv` (user ratings).
    - **Model**: The recommendation system is built using **TF-IDF** for text vectorization and **cosine similarity** for finding similar books.
    - **Framework**: Built with **Streamlit** for the user interface.
    """)

    # Sidebar - Author Info
    st.sidebar.header('Author Info')
    st.sidebar.markdown("""
    - **Name**: Your Name
    - **GitHub**: [Your GitHub Profile](https://github.com/Jyupita)
    - **Email**: kelvinkaruri@zetech.ac.ke
    """)

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
        
        st.write('### Recommended novels based on your highest rated book:')
        st.write(recommendations)
    else:
        st.sidebar.write('No books rated by this user.')

if __name__ == '__main__':
    main()