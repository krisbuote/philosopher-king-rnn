'''
Author: Kristopher Buote

Neural Network that trains on philosophy text sequences to predict the most likely next word.
Pretrained Embeddings from Stanford are used. You must first download glove.6b.100d for this code to run.
'''

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
import pickle

### LOAD AND TOKENIZE DATA SET ###
filename = 'republic_clean_sentences'
print('Loading data...')
data = open('./philosophy_data/' + filename + '.txt', 'r').read()  # should be simple plain text file
lines = data.split('\n')
print('Done.')

# Integer encoding sequences of words
tokenizer = Tokenizer()
print('Tokenizing...')
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1
print('Done.')

# Separate into input (X) and output (Y)
print("Splitting into X and Y...")
sequences = np.array(sequences)
X, Y = sequences[:,0:-1], sequences[:,-1]
# Y = to_categorical(Y, num_classes=vocab_size)
seq_length = X.shape[1]
print('Done.')

### LOAD AND PREPARE THE PRETRAINED EMBEDDING MATRIX ###
print("Loading the embeddings...")
embeddings_index = dict()
# File path to pretrained word embedding. I've used Glove.6B.100d
embedding_path = 'C:/Users/Admin/PycharmProjects/glove.6B/glove.6B.100d.txt'

with open(embedding_path, encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
print("Done.")
print("Creating the model...")
# Create Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_length,
                    weights=[embedding_matrix],trainable=False)) # Consider input_length = None
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit model
epochs = 100
for e in range(epochs):
    model.fit(X, Y, batch_size=128, epochs=1)
    # if e % 10 == 0:
    model.save('./models/philosopher-king-pretrained-epoch' + str(e) +'.h5')
    # save the tokenizer
    pickle.dump(tokenizer, open('./models/tokenizer2.pkl', 'wb'))

