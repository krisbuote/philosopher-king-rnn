import re

filename = 'republic'
data = open('./philosophy_data/' + filename + '.txt', 'r').read()  # should be simple plain text file

# turn data into clean tokens
def clean_data(data):
    clean_string = re.sub("[^a-zA-Z]",  # Remove Anything except a..z and A..Z
           " ",  # replaced with nothing
           data)  # in this string
    tokens = clean_string.split()
    tokens = [word.lower() for word in tokens]
    return tokens

tokens = clean_data(data)
print(tokens[300:500])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# organize into sequences of tokens
length = 20 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))

def save_clean_data(lines, _filename):
    clean_data = "\n".join(lines)
    with open('./philosophy_data/' + _filename + '_clean.txt', 'w') as file:
        file.write(clean_data)

save_clean_data(sequences, filename)
