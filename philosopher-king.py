'''
Author: Kristopher Buote

Neural Network that trains on philosophy text sequences to predict the most likely next word.
Embeddings are learned.
'''
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
import pickle

filename = 'republic_clean'
print('Loading data...')
data = open('./philosophy_data/' + filename + '.txt', 'r').read()  # should be simple plain text file
lines = data.split('\n')
lines = lines[0:5]
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

# Create Model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_length)) # Consider input_length = None
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit model
epochs = 1
for e in range(epochs):
    model.fit(X, Y, batch_size=512, epochs=1)
    # if e % 10 == 0:
    model.save('./models/philosopher-king-epoch' + str(e) +'.h5')
    # save the tokenizer
    pickle.dump(tokenizer, open('./models/tokenizer.pkl', 'wb'))

