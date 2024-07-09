from flask import Flask, request, render_template, redirect, url_for
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf


MODELS_DIR = "./models/"

app = Flask(__name__)

def load_model(model_file):
    """
    Load the trained model.
    """
    return joblib.load(model_file)

# Dummy function to simulate vulnerability checking
def check_vulnerability(code, model):
    # This function should implement the actual model-based checking
    # For now, we just return a dummy response
    
    # print(code) -> into the correct represenation
    
    #i need to load up the models here....

    model_file = MODELS_DIR + model + ".pkl"
    vectorize_file = MODELS_DIR + "vectorizer.pkl"

    if model != "nn":
        model = load_model(model_file)
    else:
        model = tf.keras.models.load_model(MODELS_DIR + "NN.keras")
    vectorizer = load_model(vectorize_file)

    _vector_code = vectorizer.transform([code]).toarray()
    pred = model.predict(_vector_code)
    print(pred)

    return "yolo"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    model = request.form['model']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and (file.filename.endswith('.txt') or file.filename.endswith('.c') or file.filename.endswith('.cc') or file.filename.endswith('.cpp')) :
        file_content = file.read().decode('utf-8')
        result = check_vulnerability(file_content, model)
        return render_template('index.html', result=result)
    else:
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
