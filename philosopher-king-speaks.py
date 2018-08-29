from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import random
import numpy as np

filename = 'republic_clean'
data = open('./philosophy_data/' + filename + '.txt', 'r').read()  # should be simple plain text file
lines = data.split('\n')

modelEpochNum = 99
model = load_model('./models/philosopher-king-epoch' + str(modelEpochNum) +'.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# vocab_size = model.layers[-1].output_shape[0]
seed_text = lines[np.random.randint(0, len(lines))]
seed_text = seed_text[0:-1]
print('Seed text:', seed_text)


def generate_sample(seed_text, model, tokenizer, n_words=10):
    seq_length = len(seed_text.split()) - 1

    next_word = ''
    philosopher_sample = []
    input_text = seed_text

    for l in range(n_words):
        # Text encoding
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]

        # Make sure it's only the newest words as input
        encoded_text = pad_sequences([encoded_text], maxlen=seq_length, truncating='pre')

        # Predict next word
        yhat = model.predict_classes(encoded_text, verbose=0)

        # Retrieve actual word from class
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                next_word = word
                break

        # Add word to philosopher generation sample
        philosopher_sample.append(next_word)
        # Append word to input text
        input_text += ' ' + next_word

    return " ".join(philosopher_sample)


sample = generate_sample(seed_text,model,tokenizer,n_words=7)
print(sample)