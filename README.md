# RNN_44
Text Classification using RNN
Scenario
Imagine you are working for a news aggregator platform that collects articles from various sources. To improve the user experience, the platform wants to automatically categorize these articles into relevant topics. This categorization will help users discover articles of interest more efficiently.

Objective
Your task is to develop a text classification model using Recurrent Neural Networks (RNNs) to classify these articles into predefined topics or categories. The platform has provided you with a dataset containing text articles and their corresponding labels. Your model should be able to analyze the content of these articles and assign them to the most appropriate category.

Directions
Step 1: Importing the Libraries

Import the necessary libraries using Python's import statements.
These libraries include csv, tensorflow, numpy, Tokenizer and pad_sequences from TensorFlow's Keras module, and nltk for natural language processing. matplotlib is also imported for potential data visualization.
Step 2: Defining the Hyperparameters

Set the value of vocab_size to 5000, representing the size of the vocabulary.

Set the value of embedding_dim to 64, specifying the dimensionality of the word embeddings.

Set the value of max_length to 200, indicating the maximum length of input sequences.

Set the value of padding_type to post, specifying that padding should be added at the end of sequences.

Set the value of trunc_type to post, indicating that truncation should be applied at the end of sequences.

Set the value of oov_tok to OOV, representing the token to be used for out-of-vocabulary words.

Set the value of training_portion to 0.8, representing the proportion of data to be used for training.

Step 3: Preprocessing the Data and Printing the Lengths of the Labels and Articles Lists.

Define two empty lists, articles, and labels to store the articles and labels, respectively.

Read the contents of the bbc-text.csv file using csv.reader and iterate through each row.

Extract the label from the first column of each row and append it to the labels list.

Process the article from the second column by removing stopwords and replacing consecutive spaces with a single space and then append it to the articles list.

Print the lengths of the labels and articles lists to display the number of labels and processed articles, respectively.

Step 4: Splitting the Data into Training and Validation Sets

Calculate the train_size by multiplying the length of the articles list with training_portion and converting it to an integer.

Create train_articles by slicing the articles list from index 0 to train_size.

Create train_labels by slicing the labels list from index 0 to train_size.

Create validation_articles by slicing the articles list from train_size onward.

Create validation_labels by slicing the labels list from train_size onward.

Print the train_size to display the calculated value. The lengths of train_articles, train_labels, validation_articles, and validation_labels represent the number of items in each list.

Step 5: Initializing a Tokenizer and Fitting It on the Training Articles

Initialize a Tokenizer object named tokenizer with the specified parameters: num_words representing the vocabulary size and oov_token representing the out-of-vocabulary token.

Fit the tokenizer on the training articles (train_articles) using the fit_on_texts method.

This step updates the tokenizer's internal word index based on the words in the training articles.

Assign the word index obtained from the tokenizer to the variable word_index.

Extract the first 10 items from the word_index dictionary.

Print the resulting dictionary.

Step 6: Converting the Training Articles into Sequences Using the Tokenizer

Convert the training articles (train_articles) into sequences using the texts_to_sequences method of the tokenizer object and assign the result to train_sequences.

Print the sequence representation of the 11th training article (index 10) by accessing train_sequences[10].

Step 7: Padding the Sequence

Pad the sequences in train_sequences using the pad_sequences function .
Set the maximum length of the padded sequences to max_length .
Specify the padding type as padding_type and the truncation type as trunc_type .
Assign the padded sequences to the variable train_padded.
Step 8: Printing the Length of Validation Sequences and the Shape of Validation Padded

Convert the validation articles into sequences using the tokenizer and pad the sequences to a maximum length. Assign the result to validation_padded.
Print the length of validation_sequences and the shape of validation_padded.
Create a tokenizer for the labels and fit it on the labels list.
Convert the training and validation labels into sequences using the label tokenizer and store the results in training_label_seq and validation_label_seq as NumPy arrays.
Step 9: Training the Model

Create a sequential model using tf.keras.Sequential().
Add an embedding layer to the model with the specified vocabulary size (vocab_size) and embedding dimension (embedding_dim).
Add a bidirectional LSTM layer to the model with the same embedding dimension.
Add a dense layer to the model with the embedding dimension as the number of units and relu activation function.
Add a dense layer with 6 units and the softmax activation function to the model.
Print a summary of the model's architecture using model.summary().
Step 10: Compiling the Model

Compile the model using model.compile() with the loss function set to sparse_categorical_crossentropy, the optimizer set to adam, and the metrics set to accuracy.
Set the number of epochs to 10.
Train the model using model.fit() with the training padded sequences (train_padded) and training label sequences (training_label_seq).
Specify the number of epochs as num_epochs, the validation data as the validation padded sequences (validation_padded) and validation label sequences (validation_label_seq), and verbose mode as 2.
Step 11: Plotting the Graph

Define a function named plot_graphs that takes history and string as inputs. Inside the function, plot the training and validation values of the given metric (string) from the history object using plt.plot().
Set the x-axis label as Epochs and the y-axis label as the given metric (string).
Call the plot_graphs function twice, first with history and accuracy as arguments, and then with history and loss as arguments.
Display the generated plots showing the training and validation values of the accuracy and loss metrics over the epochs.
```python
!pip install nltk
import nltk
nltk.download('stopwords')

import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
import matplotlib.pyplot as plt

```
Step 2: Defining the Hyperparameter
Set the value of vocab_size to 5000, representing the size of the vocabulary
Set the value of embedding_dim to 64, specifying the dimensionality of the word embeddings
Set the value of max_length to 200, indicating the maximum length of input sequences
Set the value of padding_type to post, specifying that padding should be added at the end of sequences
Set the value of trunc_type to post, indicating that truncation should be applied at the end of sequences
Set the value of oov_tok to OOV, representing the token to be used for out-of-vocabulary words
Set the value of training_portion to 0.8, representing the proportion of data to be used for training
```python
vocab_size = 5000
embedding_dim = 64
max_length = 200
padding_type = 'post'
trunc_type = 'post'
oov_tok = '<OOV>'
training_portion = .8
```
Step 3: Preprocessing the Data and Printing the Lengths of the Labels and Articles Lists.
Define two empty lists, articles, and labels to store the articles and labels, respectively
Read the contents of the bbc-text.csv file using csv.reader and iterate through each row
Extract the label from the first column of each row and append it to the labels list
Process the article from the second column by removing stopwords and replacing consecutive spaces with a single space and then append it to the articles list
Print the lengths of the labels and articles lists to display the number of labels and processed articles, respectively
```python
articles = []
labels = []

with open("bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)
print(len(labels))
print(len(articles))
```
Observations:

There are only 2,225 articles in the data.
Then, we split into a training set and validation set, according to the parameter we set earlier, 80% for training, and 20% for validation.
Step 4: Splitting the Data into Training and Validation Sets
Calculate the train_size by multiplying the length of the articles list with training_portion and converting it to an integer.

Create train_articles by slicing the articles list from index 0 to train_size.

Create train_labels by slicing the labels list from index** 0 to **train_size.

Create validation_articles by slicing the articles list from train_size onward.

Create validation_labels by slicing the labels list from train_size onward.

Print the train_size to display the calculated value.

The lengths of train_articles, train_labels, validation_articles, and validation_labels represent the number of items in each list.
```python
train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_articles))
print(len(train_labels))
print(len(validation_articles))
print(len(validation_labels))
```
Observations:

The value of train_size is calculated based on the proportion of training data.
The lengths of train_articles, train_labels, validation_articles, and validation_labels representing the number of items in each list.
Step 5: Initializing a Tokenizer and Fitting It on the Training Articles
Initialize a Tokenizer object named tokenizer with the specified parameters: num_words representing the vocabulary size and oov_token representing the out-of-vocabulary token.
Fit the tokenizer on the training articles (train_articles) using the fit_on_texts method.
This step updates the tokenizer's internal word index based on the words in the training articles.
Assign the word index obtained from the tokenizer to the variable word_index.
Extract the first 10 items from the word_index dictionary.
Print the resulting dictionary.
```python
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

dict(list(word_index.items())[0:10])
```
Observations:

The code prints a dictionary containing the first 10 items from the word_index dictionary.
These items represent a subset of the word-to-index mappings generated by the tokenizer.
Step 6: Converting the Training Articles into Sequences Using the Tokenizer
Convert the training articles (train_articles) into sequences using the texts_to_sequences method of the tokenizer object and assign the result to train_sequences
Print the sequence representation of the 11th training article (index 10) by accessing train_sequences[10]
```python
train_sequences  = tokenizer.texts_to_sequences(train_articles)

print(train_sequences[10])
```
Observation:

The code prints the sequence representation of the 11th training article (index 10) in the train_sequences list.
The output is a list of integers, where each integer represents the index of a word in the tokenizer's word index vocabulary that corresponds to a word in the article.
Step 7: Padding the Sequence
Pad the sequences in train_sequences using the pad_sequences function
Set the maximum length of the padded sequences to max_length
Specify the padding type as padding_type and the truncation type as trunc_type
Assign the padded sequences to the variable train_padded
```python
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(train_sequences[10])
```
Observation:

The code prints the padded sequence representation of the 11th training article.
The output is a list of integers representing the word indices of the corresponding words in the article, after applying padding to ensure a consistent length (max_length) for all sequences.
Step 8: Printing the Length of Validation Sequences and the Shape of Validation Padded
Convert the validation articles into sequences using the tokenizer and pad the sequences to a maximum length. Assign the result to validation_padded
Print the length of validation_sequences and the shape of validation_padded
Create a tokenizer for the labels and fit it on the labels list
Convert the training and validation labels into sequences using the label tokenizer and store the results in training_label_seq and validation_label_seq as NumPy arrays
```python
validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)
```
Observations:

The length of validation_sequences, indicating the number of sequences in the validation set.
The shape of validation_padded, representing the dimensions of the padded validation sequences.
```python
print(set(labels))
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
```
Observations:

The output of this code is the conversion of label sequences for the training and validation sets.
The training_label_seq and validation_label_seq are NumPy arrays containing the label sequences for the respective sets, based on the word index mapping generated by the label_tokenizer
Step 9: Training the Model
Create a sequential model using tf.keras.Sequential()
Add an embedding layer to the model with the specified vocabulary size (vocab_size) and embedding dimension (embedding_dim)
Add a bidirectional LSTM layer to the model with the same embedding dimension
Add a dense layer to the model with the embedding dimension as the number of units and relu activation function
Add a dense layer with 6 units and the softmax activation function to the model
Print a summary of the model's architecture using model.summary()
```python
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),

    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()
```
Observation:

The code outputs a summary of the model's architecture, including the number of parameters and the shape of each layer in the model.
Step 10: Compiling the Model
Compile the model using model.compile() with the loss function set to sparse_categorical_crossentropy, the optimizer set to adam, and the metrics set to accuracy
Set the number of epochs to 10
Train the model using model.fit() with the training padded sequences (train_padded) and training label sequences (training_label_seq)
Specify the number of epochs as num_epochs, the validation data as the validation padded sequences (validation_padded) and validation label sequences (validation_label_seq), and verbose mode as 2
```python
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 10

history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
```
Observations:

The code trains the model for the specified number of epochs and records the training and validation accuracy and loss metrics.
The output is an object named history that contains information about the training process, such as the loss and accuracy values at each epoch.
Step 11: Plotting the Graph
Define a function named plot_graphs that takes history and string as inputs. Inside the function, plot the training and validation values of the given metric (string) from the history object using plt.plot()
Set the x-axis label as Epochs and the y-axis label as the given metric (string)
Call the plot_graphs function twice, first with history and accuracy as arguments, and then with history and loss as arguments
Display the generated plots showing the training and validation values of the accuracy and loss metrics over the epochs
```python
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

