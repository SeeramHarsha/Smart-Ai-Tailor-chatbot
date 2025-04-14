from flask import Flask, render_template, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import re
from pymongo import MongoClient
import google.generativeai as genai
import os

# Download NLTK data for sentiment analysis
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

app = Flask(__name__)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Custom lexicon enhancements
custom_words = {
    'hell': -2.0, 'what the hell': -3.0, 'where the hell': -3.0,
    'damn': -2.0, 'wait': -1.0, 'waiting': -1.0, 'delay': -1.5,
    'delayed': -1.5, 'late': -1.5, 'where is': -1.0
}
sia.lexicon.update(custom_words)

# MongoDB setup
mongo_client = MongoClient(os.getenv('mongodb+srv://seeramharsha93:1234@cluster1.czdxp.mongodb.net/'))
db = mongo_client['ecommerce']

# Gemini setup
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

def extract_order_number(message):
    """Extract potential order number using regex pattern"""
    pattern = r'\b(?:ORD)?\d{4,8}\b'
    matches = re.findall(pattern, message.upper())
    return matches[0] if matches else None


def get_order_status(order_number):
    """Retrieve order status from MongoDB"""
    return db.orders.find_one({"order_number": order_number})

def generate_gemini_response(order_status, mood, message):
    """Generate contextual response using Gemini"""
    prompt = f"""
    company name smart ai tailor
    Customer message: {message}
    Order status: {order_status['status']}
    Customer sentiment: {mood}
    Additional info: {order_status.get('details', '')}
    
    Craft a helpful, professional response that:
    1. Acknowledges the customer's message
    2. Provides order status clearly
    3. Matches the customer's sentiment (especially if negative)
    4. Offers appropriate next steps or assistance
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Your order status: {order_status['status']}. Please let us know if you need more help."

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/api/send_message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message', '').strip()
    lower_message = message.lower()
    
    # Sentiment analysis
    sentiment = sia.polarity_scores(message)
    compound_score = sentiment['compound']
    
    # Adjust for complaint phrases
    complaint_phrases = [
        'where is my order', 'missing order', 'not received',
        'still waiting', 'haven\'t received', 'where the hell'
    ]
    if any(phrase in lower_message for phrase in complaint_phrases):
        compound_score -= 0.3
    
    # Determine mood
    mood = "neutral"
    if compound_score >= 0.05:
        mood = "positive"
    elif compound_score <= -0.05:
        mood = "negative"
    
    # Check for order number
    order_number = extract_order_number(message)
    if order_number:
        order_status = get_order_status(order_number)
        if order_status:
            response_text = generate_gemini_response(order_status, mood, message)
        else:
            response_text = f"Order {order_number} not found. Please verify the number."
        return jsonify({
            'response': response_text,
            'sentiment': mood,
            'sentiment_score': compound_score
        })
    
    # General order-related queries
    if 'order' in lower_message or 'delivery' in lower_message:
        return jsonify({
            'response': "Please provide your order number to check its status.",
            'sentiment': mood,
            'sentiment_score': compound_score
        })
    
    # General responses
    responses = {
        "positive": [
            "Thank you for your positive feedback! How can I assist you today?",
            "We're happy to hear you're satisfied! What else can we do for you?"
        ],
        "negative": [
            "I'm sorry you're having this experience. Let's resolve this quickly.",
            "Apologies for the inconvenience. Let me help fix this immediately."
        ],
        "neutral": [
            "How can I assist you today?",
            "What would you like to know more about?"
        ]
    }
    return jsonify({
        'response': random.choice(responses[mood]),
        'sentiment': mood,
        'sentiment_score': compound_score
    })

if __name__ == '__main__':
    app.run(debug=True)
