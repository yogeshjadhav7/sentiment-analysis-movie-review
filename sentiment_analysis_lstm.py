
# coding: utf-8

# In[1]:


from __future__ import print_function
import os
import cv2
import pandas as pd
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

train_file = "train_data_lstm.csv"
test_file = "test_data_lstm.csv"
MODEL_NAME = "trained_model_lstm.hdf5"

def load_data(file, direc="", sep=",", header=True):
    csv_path = os.path.join(direc, file)
    if header:
        return pd.read_csv(csv_path, sep=sep, index_col=False)
    else:
        return pd.read_csv(csv_path, sep=sep, index_col=False, header=None)
    


# In[2]:


train_data = load_data(train_file)


# In[3]:


train_data.head()


# In[4]:


test_data = load_data(test_file)


# In[5]:


test_data.head()


# In[6]:


train_labels = np.int16(train_data["Sentiment"].copy().values)
train_features = train_data.drop("Sentiment", axis=1)

test_labels = np.int16(test_data["Sentiment"].copy().values)
test_features = test_data.drop("Sentiment", axis=1)


# In[7]:


import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

maxlen = train_features.shape[1]
batch_size = 32

x_train = sequence.pad_sequences(train_features.values, maxlen=maxlen)
x_test = sequence.pad_sequences(test_features.values, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

y_train = np.array(train_labels)
y_test = np.array(test_labels)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# In[8]:


max_features = 0
train_max_features = np.max(train_features.values.flatten())
test_max_features = np.max(test_features.values.flatten())

if max_features < train_max_features:
    max_features = train_max_features

if max_features < test_max_features:
    max_features = test_max_features 
    
max_features = 2 * max_features    


# In[11]:


try:
    model = load_model(MODEL_NAME)
    print("Loaded saved model: " + MODEL_NAME)
except:
    print("Creating new model: " + MODEL_NAME)
    model = None

if model is None:
    model = Sequential()
    model.add(Embedding(max_features, 256, input_length=maxlen))
    model.add(Bidirectional(LSTM(128, dropout=0.9, recurrent_dropout=0.9)))
    model.add(Dropout(0.9))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          verbose=0,
          validation_data=[x_test, y_test],
          callbacks = [ModelCheckpoint(MODEL_NAME, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)])

saved_model = load_model(MODEL_NAME)
score = saved_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

