import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords

from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation, Dropout, Dense, Embedding, LSTM, Bidirectional, Conv1D, GRU,
    GlobalMaxPooling1D, GlobalAveragePooling1D
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Read the dataset
dataset = pd.read_csv("ORIGINAL.csv")
dataset.dropna(inplace=True)

# Function to preprocess text
def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

# Function to remove HTML tags
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

# Preprocess the text data
X = []
sentences = list(dataset['text'])
for sen in sentences:
    X.append(preprocess_text(sen))
X = pd.DataFrame({'text': X})

# Encode the labels
y = dataset['label']
le = LabelEncoder()
le.fit(y)
y = le.transform(y)

# Perform oversampling
ros = RandomOverSampler(random_state=42)
x_res, y_res = ros.fit_resample(X, y)
x_res = x_res['text']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.20, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_res)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Define vocabulary size and maximum length
vocab_size = len(tokenizer.word_index) + 1
maxlen = 100

# Pad sequences
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Load GloVe embeddings
embeddings_dictionary = dict()
with open('glove.6B.100d.txt', encoding="utf8") as glove_file:
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
        
embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# Build and compile the model
model = Sequential([
    Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False),
    Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
    Bidirectional(LSTM(54, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
    Bidirectional(LSTM(60, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation="relu"),
    Dense(6, activation="softmax")
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, batch_size=128, epochs=30, verbose=1, validation_split=0.2)

# Save the model and preprocessing objects
model.save("model_emotion_new.hdf5")

with open('label_emotion.pickle', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('tokenizer_emotion.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Visualize training history
import matplotlib.pyplot as plt

plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()
