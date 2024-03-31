# preprocessing.py

import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to load enhanced annotations from a JSON file
def load_enhanced_annotations(filepath):
    """
    Load enhanced annotations from a JSON file.

    :param filepath: Path to the JSON file containing enhanced annotations.
    :return: Dictionary with loaded enhanced annotations.
    """
    with open(filepath, 'r') as file:
        return json.load(file)

# Function to prepare the dataset for LSTM model training
def prepare_dataset_for_lstm(annotations, num_words=10000, max_seq_length=None):
    """
    Prepare the dataset for LSTM training.

    :param annotations: Enhanced annotations including both vulnerable and non-vulnerable lines.
    :param num_words: Number of top words to keep in the tokenizer.
    :param max_seq_length: Maximum length of sequences. If None, it will be set based on the data.
    :return: Tuple containing training and test data (X_train, X_test, y_train, y_test).
    """
    tokenizer = Tokenizer(num_words=num_words)
    contexts = [sample['context'] for filename, annotations in annotations.items() for line_num, sample in annotations.items()]
    tokenizer.fit_on_texts(contexts)

    sequences = tokenizer.texts_to_sequences(contexts)
    if not max_seq_length:
        max_seq_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length)

    labels = np.array([sample['is_vulnerable'] for filename, annotations in annotations.items() for line_num, sample in annotations.items()])

    # Assuming an 80-20 train-test split for demonstration purposes
    split_index = int(len(padded_sequences) * 0.8)
    X_train, X_test = padded_sequences[:split_index], padded_sequences[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]

    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    # Assuming the enhanced_annotations.json is in the same directory
    enhanced_annotations = load_enhanced_annotations('enhanced_annotations.json')
    X_train, X_test, y_train, y_test = prepare_dataset_for_lstm(enhanced_annotations)
s