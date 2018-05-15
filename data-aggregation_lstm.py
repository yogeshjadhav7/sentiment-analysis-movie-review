
# coding: utf-8

# In[1]:


#Import all the dependencies
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import numpy as np

import pandas as pd

PRUNE = False
POS_TRAIN_PATH = "aclImdb/train/pos/"
NEG_TRAIN_PATH = "aclImdb/train/neg/"
POS_TEST_PATH = "aclImdb/test/pos/"
NEG_TEST_PATH = "aclImdb/test/neg/"


# In[2]:


train_features = []
train_labels = []
test_features = []
test_labels = []


# In[3]:


filenames = [f for f in listdir(POS_TRAIN_PATH) if f.endswith('.txt')]
for filename in filenames:
    data = open(POS_TRAIN_PATH + filename).read()
    train_features.append(data)
    train_labels.append(1)
    
filenames = [f for f in listdir(NEG_TRAIN_PATH) if f.endswith('.txt')]
for filename in filenames:
    data = open(NEG_TRAIN_PATH + filename).read()
    train_features.append(data)
    train_labels.append(0)


# In[4]:


filenames = [f for f in listdir(POS_TEST_PATH) if f.endswith('.txt')]
for filename in filenames:
    data = open(POS_TEST_PATH + filename).read()
    test_features.append(data)
    test_labels.append(1)
    
filenames = [f for f in listdir(NEG_TEST_PATH) if f.endswith('.txt')]
for filename in filenames:
    data = open(NEG_TEST_PATH + filename).read()
    test_features.append(data)
    test_labels.append(0)


# In[5]:


tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
        
   return new_data

train_features = nlp_clean(train_features)
test_features = nlp_clean(test_features)


# In[6]:


mean_len = 0
bow = []
for f in train_features:
    mean_len = mean_len + len(f)
    bow.extend(f)
    
for f in test_features:
    mean_len = mean_len + len(f)
    bow.extend(f)
    
mean_len = np.int32(mean_len / (len(bow)))
bow = np.array(bow)


# In[7]:


mean_len = np.int16(np.ceil(mean_len / 100) * 100)


# In[8]:


mean_len


# In[9]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(bow)
print(len(encoder.classes_))


# In[10]:


x_train = []
x_test = []

counter = 0
for f in train_features:
    r = np.array(f)
    r = encoder.transform(r)
    x_train.append(r)
    counter = counter + 1
    if counter % 1000 == 0:
        print(counter)
        if PRUNE:
            break

counter = 0
for f in test_features:
    r = np.array(f)
    r = encoder.transform(r)
    x_test.append(r)
    counter = counter + 1
    if counter % 1000 == 0:
        print(counter)
        if PRUNE:
            break


# In[11]:


x_train_copy = x_train.copy()
x_test_copy = x_test.copy()


# In[12]:


from keras.preprocessing import sequence

x_train = sequence.pad_sequences(x_train_copy, maxlen=mean_len)
x_test = sequence.pad_sequences(x_test_copy, maxlen=mean_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# In[13]:


columns = [str(x) for x in range(mean_len)]
columns.append("Sentiment")


# In[14]:


train_data = None

for idx in range(len(x_train)):
    features = x_train[idx]
    label = np.array([train_labels[idx]], dtype=np.int16)
    row = np.array([np.concatenate((features, label), axis = 0)])
    
    if train_data is None:
        train_data = row
    else:
        train_data = np.concatenate((train_data, row), axis=0)
    


# In[15]:


np.shape(train_data)


# In[16]:


test_data = None

for idx in range(len(x_test)):
    features = x_test[idx]
    label = np.array([test_labels[idx]], dtype=np.int16)
    row = np.array([np.concatenate((features, label), axis = 0)])
    
    if test_data is None:
        test_data = row
    else:
        test_data = np.concatenate((test_data, row), axis=0)
        


# In[17]:


np.shape(test_data)


# In[18]:


train_data = pd.DataFrame(columns=columns, data=train_data)
test_data = pd.DataFrame(columns=columns, data=test_data)


# In[19]:


train_data = train_data.sample(frac=1)
test_data = test_data.sample(frac=1)


# In[20]:


train_data.head()


# In[21]:


train_data.info()


# In[22]:


test_data.head()


# In[23]:


test_data.info()


# In[24]:


train_data.to_csv("train_data_lstm.csv", index=False)
test_data.to_csv("test_data_lstm.csv", index=False)

