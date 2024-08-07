# Code Vulnerability Detection AI: Function Explanations

This document provides detailed explanations of each function within the Python script designed for detecting vulnerabilities in code. The script is decomposed into functions for better readability, maintainability, and educational purposes.

## 1. `load_and_inspect_data(file_path)`

### Purpose:
This function is responsible for loading the dataset from a CSV file and displaying its basic information and the first few rows. This initial inspection is crucial for understanding the dataset's structure, which informs subsequent preprocessing steps.

### How It Works:
- The function takes `file_path` as an argument, which specifies the location of the dataset.
- It uses `pandas` to read the CSV file and store it in a DataFrame.
- The `.info()` and `.head()` methods are then used to print the dataset's structure and preview the first few entries, respectively.

### Why This Way:
Loading and inspecting the data first provides a quick overview of the dataset's content, types of data it contains, and potential issues with missing values or data inconsistency. This step is fundamental in data analysis and preprocessing pipelines.

## 2. `clean_data(dataset)`

### Purpose:
To clean the dataset by removing rows with missing values and duplicates. This ensures the model trains on high-quality, unique data points.

### How It Works:
- The function accepts a DataFrame and applies the `.dropna()` method to remove any rows with missing data.
- It then uses `.drop_duplicates()` to ensure all data points are unique.

### Why This Way:
Data cleaning is essential to remove noise and prevent the model from learning from incomplete or redundant data. Removing duplicates helps in reducing overfitting and computational redundancy during training.

## 3. `preprocess_data(dataset)`

### Purpose:
Prepares the dataset for modeling by tokenizing the source code and applying one-hot encoding to the vulnerability types.

### How It Works:
- Initializes a `Tokenizer` object from Keras, which is configured to only consider the top 10,000 words (or another threshold based on the dataset).
- The source code is then tokenized, converting each code snippet into a sequence of integers.
- These sequences are padded to have the same length, which is necessary for batch processing in neural networks.
- One-hot encoding transforms the categorical vulnerability types into a binary matrix representation, suitable for model training.

### Why This Way:
Tokenization converts text data into a format that neural networks can work with, while padding ensures consistent input sizes. One-hot encoding is a standard approach for handling categorical labels in machine learning, enabling the model to predict the probability of each category.

## 4. `build_model(tokenizer, max_length, output_dim)`

### Purpose:
Defines and compiles a neural network model for the task, using an LSTM layer suited for sequence data like code.

### How It Works:
- The model uses an `Embedding` layer to convert tokenized integers into dense vectors of fixed size.
- An `LSTM` layer is used to process these embeddings, capturing the sequential dependencies in the code.
- The `Dense` layer at the end, with a softmax activation, outputs the probability distribution over different vulnerability types.
- The model is compiled with the 'adam' optimizer and 'categorical_crossentropy' loss function, appropriate for multi-class classification.

### Why This Way:
This architecture leverages the sequential nature of LSTM to process code effectively, learning patterns across the code snippets. The embedding layer allows the model to learn an efficient representation of tokens, crucial for understanding the context and semantics in the code.

## 5. `train_and_evaluate_model(model, dataset)`

### Purpose:
Trains the compiled model on the preprocessed dataset and evaluates its performance on a test set.

### How It Works:
- The function converts the tokenized and padded code sequences, along with their one-hot encoded labels, into numpy arrays for training.
- It splits the data into training and testing sets to evaluate the model's performance on unseen data.
- The model is then trained using the training data, with validation performed on a subset of this data.
- Finally, the model's accuracy is evaluated on the test set.

### Why This Way:
Training and evaluation are split into distinct steps to assess the model's ability to generalize beyond the data it was trained on. The use of a test set helps in evaluating the model's performance in a real-world scenario, ensuring it has not just memorized the training data.

## Main Execution Flow: `main()`

### Purpose:
Orchestrates the execution of all steps in the pipeline when the script is run directly.

### How It Works:
- Sequentially calls the functions defined above, passing the necessary data between them.
- Begins with loading and cleaning the data






# How did we find the line? 

### Our Model

So, our model in general is able to tell us whether or not a code file has a vulnerability present! It just says, "yes you do!" or "No you dont!". Ok, it would be too difficult (too much computation resources) to then train another model to tell us which lines are the issue. 


Well, the model should already "have" an understanding of "which code" is problematic if it already has the ability to tell us if the file is vulnerable or not. 


### Our Solution
We removed each each line, one by one, and then re-asked the model if it thought there was still a vulnerability. 

If the model changed it's mind or had a large "difference" in confidance just by removing the line, then we knew that this was a super important line for the prediction. 

### Overfitting Issues

By doing this, we found something troubling. The model always thought that the first line was the most important. We saw that the first line was always a comment (e.g. "//FormAI DATASET v1.0 Category: Network Topology Mapper ; Style: multi-threaded"). So the model started to associate "vulnerability" with these comments. Once, we removed this line from a test-file, the model correctly showed us that this line: 25 (scanf(""%s"", user);) had the possibility of being buffer overflowed. 

Also, we found that silly comments also gravely influence the models ability to make predictions. So we will be removing All comments from code and will retrain all models!

# WebApp Features 

### Display the Text File in a nice IDE. 

### Displaying Vulnerable Lines

Tell the user which line is the most indicative of the error and we can also highlight it in the Text File we render ABOVE! 

We use a sliding scale highlight, showing the user which lines are the most important just in case there's more than one. 

## What if we fix the code for them...or we suggest a possible solution. And this doesn't have to be AI!

"Ok, we detect that the issue comes from the "strcspn" function, a common fix would be to do ......" 












