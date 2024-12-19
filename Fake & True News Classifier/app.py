from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string 
from nltk.corpus import stopwords

from flask import Flask, request, render_template
import joblib

# Load the plot model and vectorizer
model = joblib.load('SVC_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict',  methods=['POST'])
def predict():
    # Get the input text from the user
    if request.method == 'POST':
      new_article = request.form['news_article']

      # Preprocess the input text
      processed_text = transform(new_article) # Use the preprocess_text function

      # Transform the input text using the loaded vectorizer
      vectorizer_input = vectorizer.transform([processed_text]).toarray()

      # Predict using the loaded model
      prediction = model.predict(vectorizer_input)

      # Map the prediction to label
      label = 'True' if prediction[0] == 1 else 'Fake'

      return render_template('index.html', prediction_text=f'The news article is predicted to be: {label}')


lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def transform(text):
    text = text.lower()  # Convert text to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    # Lemmatize, remove stopwords and punctuation
    tokens = [lemmatizer.lemmatize(word, "v") for word in tokens if word not in stopwords and word not in punctuations]
    return " ".join(tokens)

if __name__ == '__main__':
   app.run(debug=True)