import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 

def load_and_inspect_data(file_path):
    """
    Load and inspect the initial dataset.
    Note: We return a Pandas Dataframe.
    """
    dataset = pd.read_csv(file_path)
    # print(dataset.info()) 1 2 3 ayyyy
    # print(dataset.head(1))
    return dataset

def clean_data(dataset):
    """
    Perform data cleaning, including removing duplicates and handling missing values.
    """
    print("Cleaning the data!")    
    dataset_cleaned = dataset.drop_duplicates()
    return dataset_cleaned

def make_bow_represenations(dataset): 
    '''
    "the man at the burrito" ->  ['the', 'man', 'ate', 'the', 'burrito']

      'the',  -> [1,0,0,0]
      'man',  -> [0,1,0,0]    
      'ate',  -> [0,0,1,0]
      'the',  -> [1,0,0,0]
      'burrito'-> [0,0,0,1]
    
     "the man at the burrito" -> [2,1,1,1]
     "man at the burrito the" -> 
     "at the man the burrito" -> 
     Word order is lost and so it 

     
     85,000 words in my vocab

     if(X == three) reutrn X

     if X three return X
    
     by a 85,000 long vector with only 4 non zero values
     '''
    vectorize = CountVectorizer(max_features = 1000)

    just_coding_col= dataset["Source code"]
    X = vectorize.fit(just_coding_col) #we only use the top 1000 most used tokens
    #why would we want to decrease the number of tokens we look at? 
    print(X.vocabulary_)
    X = vectorize.transform(just_coding_col)
    np.save("bow_array", X.toarray())
    
    X_bow =  np.load("bow_array.npy")
    y_bow =  np.load("y.npy")

    return X_bow, y_bow
    #we have the features loaded up! "The code part" 

    #We also need one more thing to finaly train a model.
    '''
    X -> Code
    y -> What is Vulnerable and what isn't?
    '''

    # dataset["y"] = np.where(dataset['Vulnerability type'] == "NOT VULNERABLE up to bound k", 0, 1)

    # # Print the new column to check
    # #The amount of code with vulnerabilities? 
    # np.save("y", dataset["y"])

def preprocess_data(dataset):
    """
    Tokenize source code and apply one-hot encoding to the vulnerability type.

    Examples: 
    "The man ate the burrito" -> ['the', 'man', 'ate', 'the', 'burrito']
    ['the', 'man', 'ate', 'the', 'burrito'] -> Embedding Space ([100, 101, 102, 101, 1]) 

    Each word maps to a unique vector of numebrs! 


    
    #one hot encodings! 
    ['the', 'man', 'ate', 'the', 'burrito'] ->
    [
        [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [1,0,0,0],
            [0,0,0,1]
        ]
    ]
    Vocabulary: 4, the man ate burrito
    """

    print("Tokenizing the data!")    
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(dataset['Source code']) #update the vocabulary of the vectorizer! 

    dataset['Code_Tokens'] = tokenizer.texts_to_sequences(dataset['Source code'])

    #is going to grab the largest value in the list, which is a 
    #list of sequnce lengths. 

    max_length = max([len(seq) for seq in dataset['Code_Tokens']])

    '''

    _list = []
    for seq in dataset['Code_Tokens']: 
        _list.append(len(seq))
    max_length = max(_list)

    Examples of a list of sequences (pre numbers)
    [
        [ 
            "hi", "how", "are", "you"
        ]
        [
            "goodbye", 'loser'
        ]
    ]
    '''

    dataset['Code_Tokens_Padded'] = pad_sequences(dataset['Code_Tokens'], maxlen=max_length, padding='post').tolist()

    print("Creating one hot encodings of Y!") 
    ohe = OneHotEncoder(sparse=False)
    dataset['label_OHE'] = list(ohe.fit_transform(dataset[['Error type']]))

    return dataset, tokenizer, max_length

def build_model(tokenizer, max_length, output_dim):
    """
    Build and compile the LSTM model.
    """
    print("Buidling model!") 
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50

    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))

    model.add(LSTM(128, return_sequences=False))
    #enter each token into the list one at a time
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model

def train_and_evaluate_model(model, dataset):
    """
    Train the model and evaluate its performance.
    """
    print('training model!')
    X = np.array(dataset['Code_Tokens_Padded'].tolist())
    y = np.array(dataset['label_OHE'].tolist())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')

def main():
    """
    Main function to execute the steps.

    CSV LOOKS LIKE THIS: Filename, Vulnerability type, Source code, Function name, Line, Error type
    """

    file_path = 'FormAI_dataset.csv'
    dataset = load_and_inspect_data(file_path) #we have loaded the dataset! 
    dataset_cleaned = clean_data(dataset)
    X, y = make_bow_represenations(dataset_cleaned)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    #we sorta went crazy we went straight to the "best case" solution
    #Embeddings and RNNs as well. 
    # dataset_preprocessed, tokenizer, _max_length = preprocess_data(dataset_cleaned)
    return 
    model = build_model(tokenizer, max_length=_max_length, output_dim=len(dataset_preprocessed.iloc[0]['label_OHE']))
    train_and_evaluate_model(model, dataset_preprocessed)

if __name__ == "__main__":
    main()
