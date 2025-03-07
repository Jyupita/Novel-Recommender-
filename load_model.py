import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Enabling  logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the data we need
books = pd.read_csv("C:/Users/kl/Desktop/Novel Recommender/books_cleaned.csv")
ratings = pd.read_csv("C:/Users/kl/Desktop/Novel Recommender/ratings.csv")

# Compute TF-IDF matrix for book titles
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(books["title"])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# The recommendation function
def get_recommendations(title):
    if title not in books["title"].values:
        return ["Sorry, book not found in the database. Try another title."]
    
    idx = books[books["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get top 5 recommendations
    book_indices = [i[0] for i in sim_scores]
    
    return books["title"].iloc[book_indices].tolist()

# This is the function to handle recommendations in the Telegram bot
async def recommend(update: Update, context: CallbackContext):
    user_input = update.message.text.strip()  #This will get the book title from the user (the user to input the book title)
    recommended_books = get_recommendations(user_input)

    response = f"📚 *Recommended books based on '{user_input}':*\n\n"
    response += "\n".join([f"✅ {book}" for book in recommended_books])

    await update.message.reply_text(response, parse_mode="Markdown")

# Start command function for the bot
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("📖 Welcome to the Novel Recommendation Bot!\n\nSend me a book title, and I'll recommend similar books for you.")

#  This is the main function to run the bot using the token
def main():
    TOKEN = "8156559476:AAGo75-zEUERpaYSLzxNXxeaxeOCmoDrc5k"

    app = Application.builder().token(TOKEN).build()

    # Adding handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, recommend))

    # Start the bot
    logger.info("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()



