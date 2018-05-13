
# coding: utf-8

# In[1]:


import os
import cv2
import pandas as pd
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

train_file = "train_data.csv"
test_file = "test_data.csv"

TRAIN_MODEL = True
MODEL_NAME = "trained_model_mlp.hdf5"

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


from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
scalar.fit(train_features)

train_features = scalar.transform(train_features)
test_features = scalar.transform(test_features)


# In[8]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
def plot_roc_curve(clf_sets):
    for clf_set in clf_sets:
        y = clf_set[0]
        y_pred = clf_set[1]
        label = clf_set[2]
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        plt.plot(fpr, tpr, linewidth=1, label=label)
    
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="bottom right")
    plt.show()


# In[9]:


X = train_features.copy()
Y = train_labels.copy()
X_test = test_features.copy()
Y_test = test_labels.copy()


# In[10]:



# CNN Classifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

batch_size = 32
epochs = 25

size = np.int16(X.shape[1])

train_x = X.copy()
test_x = X_test.copy()

train_y = to_categorical(Y)
test_y = to_categorical(Y_test)

num_classes = train_y.shape[1]
droprate = 0.8

try:
    model = load_model(MODEL_NAME)
except:
    model = None
    
ACT = 'tanh'    
    
if model is None:
    model = Sequential()

    model.add(Dense(1024, activation=ACT, input_shape=(size,)))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))

    model.add(Dense(512, activation=ACT))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))
    
    model.add(Dense(512, activation=ACT))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))

    model.add(Dense(256, activation=ACT))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))
    
    model.add(Dense(128, activation=ACT))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))
    
    model.add(Dense(64, activation=ACT))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))
    
    model.add(Dense(32, activation=ACT))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))
    
    model.add(Dense(16, activation=ACT))
    model.add(BatchNormalization())
    model.add(Dropout(droprate))

    model.add(Dense(num_classes, activation='softmax'))

    adam = Adam()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
else:
    print(MODEL_NAME, " is restored.")

model.summary()

callbacks = [ModelCheckpoint(MODEL_NAME, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)]

if TRAIN_MODEL:
    history = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(test_x, test_y),
                        callbacks=callbacks)
else:
    print("Opted not to train the model as TRAIN_MODEL is set to False. May be because model is already trained and is now being used for validation")
    


# In[ ]:


saved_model = load_model(MODEL_NAME)
score = saved_model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


test_x.shape

