
'''
Once the model tells us that it is a piece of vulnerable code, we will do the following. 

We will prompt the model with version of the code with lines missing. And if the model changes its prediction from Vulnerable to Unvulnerable 
we will assume (this is important!) that these were the lines which were the most faulty. This will allow us to present these 
lines to the user as being vulnerable (or containing the vulnerability)
'''


import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam 

import tensorflow as tf
import joblib

MODELS_DIR = "./server/models/"
TEST_CODE = "./test_code/vul.c"

model_file = MODELS_DIR + "nn" + ".pkl"
vectorize_file = MODELS_DIR + "vectorizer.pkl"

def load_model(model_file):
    """
    Load the trained model.
    """
    return joblib.load(model_file)



model = load_model(model_file)
    # model = tf.keras.models.load_model(MODELS_DIR + "NN.keras")
vectorizer = load_model(vectorize_file)


# Dummy function to simulate vulnerability checking
def check_vulnerability(code):
    # This function should implement the actual model-based checking
    # For now, we just return a dummy response
    
    # print(code) -> into the correct represenation
    
    #i need to load up the models here....
    _vector_code = vectorizer.transform([code]).toarray()
    pred = model.predict(_vector_code)
    return pred
    

def load_test_code(file_path):
    """
    Load the test code from a file.
    """

    #READS IN THE ENTIRE FILE. is that what we want? 
    with open(file_path, 'r') as file:
        code = file.read()
    return code



def test_each_line():

    entire_code = load_test_code(TEST_CODE)
    our_solid_truth = check_vulnerability(entire_code)
    fr_conf = our_solid_truth[0][1]
    print(fr_conf) # 9.9965882e-01]

    _code_split = load_test_code(TEST_CODE).split("\n")
    len_code = len(_code_split)
    _list_of_diffs = [] 
    for i in range(len_code-1): #This will loop for every line! 
        
        print(f"removing line {i}")
        to_test = "\n".join(_code_split[0:i] + _code_split[i+1:])
        res = check_vulnerability(to_test)
        current_value = res[0][1]
        diff = fr_conf - current_value
        _list_of_diffs.append(diff)
        print(f"Predicted {np.argmax(res)}")
    print("MOST IMPROTANT LINE IS: ", np.argmax(_list_of_diffs) + 1)


test_each_line()

# check_vulnerability(todo, "nn")