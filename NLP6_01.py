"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Source: https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

################################## Loat Text ##################################
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
# load text
raw_text = load_doc('C:\\DataScience\\DataSet\\rhyme.txt')
raw_text = raw_text.lower()
print(raw_text)


################################## Clean Text #################################
tokens = raw_text.split()
raw_text = ' '.join(tokens)


################################ Create Sequences #############################
length = 10
sequences = list()
for i in range(length, len(raw_text)):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	# store
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))


################################# Save Sequence ###############################
# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()    
# save sequences to file
out_filename = 'C:\\DataScience\\DataSet\\char_sequences.txt'
save_doc(sequences, out_filename)


############################## Train Language Model ###########################
# Load Data
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text 
# load
in_filename = 'C:\\DataScience\\DataSet\\char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

# Encode Sequences
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)
    
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# Split Inputs and Output
sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=100, verbose=2)
 
# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))


################################# Generate Text ###############################
# Generate Text
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		# encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text

# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))

# test start of rhyme
print(generate_seq(model, mapping, 10, 'Sing a son', 21))
# test mid-line
print(generate_seq(model, mapping, 10, 'king was i', 20))
# test not in original
print(generate_seq(model, mapping, 10, 'hello worl', 20))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Dataset: Alice in wonderland
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

################################## Loat Text ##################################
in_filename = 'C:\\DataScience\\DataSet\\AliceInWonderland.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

################################## Clean Text #################################
tokens = raw_text.split()
raw_text = ' '.join(tokens)

################################ Create Sequences #############################
length = 10
sequences = list()
for i in range(length, len(raw_text)):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	# store
	sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

################################# Save Sequence ###############################
# save tokens to file, one dialog per line   
# save sequences to file
out_filename = 'C:\\DataScience\\DataSet\\char_sequences.txt'
save_doc(sequences, out_filename)

############################## Train Language Model ###########################
# Load Data
in_filename = 'C:\\DataScience\\DataSet\\char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

# Encode Sequences
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)
    
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# Split Inputs and Output
sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=300, verbose=2)
 
# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))

################################# Generate Text ###############################
# Generate Text
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))

# test start of rhyme
print(generate_seq(model, mapping, 10, 'Alice was beginning', 25))
# test mid-line
print(generate_seq(model, mapping, 10, 'golden key', 30))
# test not in original
print(generate_seq(model, mapping, 10, 'hello worl', 20))




