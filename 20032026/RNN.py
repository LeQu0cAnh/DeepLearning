# import numpy
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.utils import np_utils
# # load text
# filename = "christmasStories.txt"

# text = (open(filename).read()).lower()

# # mapping characters with integers
# unique_chars = sorted(list(set(text)))
# print(unique_chars)

# char_to_int = {}
# int_to_char = {}

# for i, c in enumerate (unique_chars):
#     char_to_int.update({c: i})
#     int_to_char.update({i: c})
#     # preparing input and output dataset
# X = []
# Y = []

# for i in range(0, len(text) - 50, 1):
#     sequence = text[i:i + 50]
#     label =text[i + 50]
#     #print(sequence)
#     X.append([char_to_int[char] for char in sequence])
#     Y.append(char_to_int[label])
# #print(Y)
# # reshaping, normalizing and one hot encoding
# X_modified = numpy.reshape(X, (len(X), 50, 1))
# X_modified = X_modified / float(len(unique_chars))
# Y_modified = np_utils.to_categorical(Y)
# # defining the LSTM model
# model = Sequential()
# model.add(LSTM(300, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(300))
# model.add(Dropout(0.2))
# model.add(Dense(Y_modified.shape[1], activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam')
# # fitting the model
# model.fit(X_modified, Y_modified, epochs=1, batch_size=30)

# # picking a random seed
# start_index = numpy.random.randint(0, len(X)-1)
# new_string = X[start_index]

# # generating characters
# for i in range(50):
#     x = numpy.reshape(new_string, (1, len(new_string), 1))
#     x = x / float(len(unique_chars))

#     #predicting
#     pred_index = numpy.argmax(model.predict(x, verbose=0))
#     char_out = int_to_char[pred_index]
#     seq_in = [int_to_char[value] for value in new_string]
#     print(char_out)

#     new_string.append(pred_index)
#     #remove the first character
#     new_string = new_string[1:len(new_string)]
import numpy
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from keras.utils import to_categorical

# 1. Load text
filename = "christmasStories.txt"
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read().lower()

# 2. Mapping
unique_chars = sorted(list(set(text)))
n_vocab = len(unique_chars)
char_to_int = {c: i for i, c in enumerate(unique_chars)}
int_to_char = {i: c for i, c in enumerate(unique_chars)}

# 3. Dataset
X = []
Y = []
seq_length = 50
for i in range(0, len(text) - seq_length, 1):
    X.append([char_to_int[char] for char in text[i:i + seq_length]])
    Y.append(char_to_int[text[i + seq_length]])

# 4. Reshape
X_modified = numpy.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(n_vocab)
Y_modified = to_categorical(Y)

# 5. Model
model = Sequential()
model.add(Input(shape=(seq_length, 1)))
model.add(LSTM(300, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# 6. Train (Để epochs=1 và batch_size=128 để khớp với mẫu của bạn)
model.fit(X_modified, Y_modified, epochs=1, batch_size=128)

# 7. Generate (In dọc từng chữ)
start_index = numpy.random.randint(0, len(X)-1)
new_string = X[start_index]

for i in range(50):
    x = numpy.reshape(new_string, (1, len(new_string), 1))
    x = x / float(n_vocab)
    
    pred_index = numpy.argmax(model.predict(x, verbose=0))
    char_out = int_to_char[pred_index]
    
    # ĐÁP SỐ BẠN MUỐN: In từng ký tự trên 1 dòng
    print(char_out)

    new_string.append(pred_index)
    new_string = new_string[1:]