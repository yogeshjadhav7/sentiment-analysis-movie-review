
# coding: utf-8

# In[1]:


#Import all the dependencies
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import numpy as np

import pandas as pd

MODEL_NAME = "doc2vec.model"
SIZE = 25
WINDOW_SIZE = 3

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


# In[6]:


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


class LabeledLineSentence(object):
    
    def __init__(self, docs):
        self.docs = docs
        
    def __iter__(self):
        for idx, doc in enumerate(self.docs):
              yield gensim.models.doc2vec.LabeledSentence(doc,[str(idx)])

iterator = LabeledLineSentence(train_features)
model = gensim.models.Doc2Vec(size=SIZE * SIZE, window=WINDOW_SIZE, min_count=5, workers=16,alpha=0.025, min_alpha=0.025, iter=25)
model.build_vocab(iterator)
model.train(iterator, epochs=model.iter, total_examples=model.corpus_count)

model.save(MODEL_NAME)
print(MODEL_NAME + " saved")


# In[7]:


columns = [str(x) for x in range(len(model.docvecs[1]))]
columns.append("Sentiment")


# In[8]:


train_data = None

for idx in range(len(train_labels)):
    features = np.array(model.docvecs[str(idx)])
    label = np.array([train_labels[idx]], dtype=np.int16)
    row = np.array([np.concatenate((features, label), axis = 0)])
    
    if train_data is None:
        train_data = row
    else:
        train_data = np.concatenate((train_data, row), axis=0)
    


# In[9]:


np.shape(train_data)


# In[10]:


test_data = None

for idx in range(len(test_labels)):
    features = np.array(model.infer_vector(test_features[idx]))
    label = np.array([test_labels[idx]], dtype=np.int16)
    row = np.array([np.concatenate((features, label), axis = 0)])
    
    if test_data is None:
        test_data = row
    else:
        test_data = np.concatenate((test_data, row), axis=0)
        


# In[11]:


np.shape(test_data)


# In[12]:


train_data = pd.DataFrame(columns=columns, data=train_data)
test_data = pd.DataFrame(columns=columns, data=test_data)


# In[13]:


train_data = train_data.sample(frac=1)
test_data = test_data.sample(frac=1)


# In[14]:


train_data.head()


# In[15]:


train_data.info()


# In[16]:


test_data.head()


# In[17]:


test_data.info()


# In[18]:


train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

