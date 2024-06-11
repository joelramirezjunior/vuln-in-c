import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import joblib

def load_and_inspect_data(file_path):
    """
    Load and inspect the initial dataset.
    """
    dataset = pd.read_csv(file_path)
    return dataset

def clean_data(dataset):
    """
    Perform data cleaning, including removing duplicates.
    """
    dataset_cleaned = dataset.drop_duplicates()
    return dataset_cleaned

def make_bow_representations(dataset): 
    """
    Create Bag of Words (BoW) representations of the source code.
    """
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(dataset["Source code"]).toarray()
    y = np.where(dataset['Vulnerability type'] == "NOT VULNERABLE up to bound k", 0, 1)
    
    return X, y, vectorizer

def save_transformed_data(X, y, vectorizer, X_file, y_file, vectorizer_file):
    """
    Save the transformed data and vectorizer.
    """
    np.save(X_file, X)
    np.save(y_file, y)
    joblib.dump(vectorizer, vectorizer_file)

def load_transformed_data(X_file, y_file, vectorizer_file):
    """
    Load the transformed data and vectorizer.
    """
    X = np.load(X_file)
    y = np.load(y_file)
    vectorizer = joblib.load(vectorizer_file)
    return X, y, vectorizer

def save_model(model, model_file):
    """
    Save the trained model.
    """
    joblib.dump(model, model_file)

def load_model(model_file):
    """
    Load the trained model.
    """
    return joblib.load(model_file)

def load_test_code(file_path):
    """
    Load the test code from a file.
    """
    with open(file_path, 'r') as file:
        code = file.read()
    return code

def main():
    """
    Main function to execute the steps.
    """
    file_path = 'FormAI_dataset.csv'
    X_file = 'X.npy'
    y_file = 'y.npy'
    vectorizer_file = 'vectorizer.pkl'
    model_file = 'knn_model.pkl'
    
    if os.path.exists(X_file) and os.path.exists(y_file) and os.path.exists(vectorizer_file):
        X, y, vectorizer = load_transformed_data(X_file, y_file, vectorizer_file)
    else:
        dataset = load_and_inspect_data(file_path)
        dataset_cleaned = clean_data(dataset)
        X, y, vectorizer = make_bow_representations(dataset_cleaned)
        save_transformed_data(X, y, vectorizer, X_file, y_file, vectorizer_file)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        save_model(model, model_file)
    
    accuracy = model.score(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')
    
    # Load and test the model with new code files
    test_files = ['test_code/non_vul.c', 'test_code/vul.c']
    for file in test_files:
        test_code = load_test_code(file)
        test_vector = vectorizer.transform([test_code]).toarray()
        prediction = model.predict(test_vector)
        print(f'File: {file}, Prediction: {"VULNERABLE" if prediction[0] == 1 else "NOT VULNERABLE"}')

if __name__ == "__main__":
    main()