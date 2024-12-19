from flask import Flask, request, render_template
from tensorflow.keras.models import load_model                                                                                                    # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences                                                                                          # type: ignore
import pickle

# Load the trained model
model = load_model('sentiment_analysis_model.keras')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Maximum length of a review
maxlen = 200

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the review from the form
        review = request.form['review']
        
        # Preprocess and tokenize the review
        review_seq = tokenizer.texts_to_sequences([review])
        review_padded = pad_sequences(review_seq, maxlen=maxlen, padding='post')
        
        # Predict sentiment
        prediction = model.predict(review_padded)[0][0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        
        return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
