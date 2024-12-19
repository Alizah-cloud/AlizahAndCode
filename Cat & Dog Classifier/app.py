from flask import Flask, render_template, request                                                                                                                                
import tensorflow as tf                                                                                                                                                                    
from tensorflow.keras.preprocessing.image import img_to_array                                                                                                                           
from PIL import Image                                                                                                                                                                      
import numpy as np                                                                                                                                                                                                                                                      

# Initialize Flask app
app = Flask(__name__)

model = tf.keras.models.load_model('classification_model.keras')


def predict_image(image):
    img = image.resize((128, 128)) 
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    prediction = model.predict(img_array) 
    return "Dog" if prediction[0][0] > 0.5 else "Cat"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file') 
        if not file:
            return render_template('index.html', error="No file uploaded!")
        try:
            image = Image.open(file) 
            prediction = predict_image(image)  
            return render_template('index.html', prediction=prediction)
        except Exception as e:
            return render_template('index.html', error=f"Error processing image: {str(e)}")
    return render_template('index.html') 

if __name__ == '__main__':
    app.run(debug=True)
