
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import keras
from keras.utils import plot_model
from keras import regularizers
import keras.layers as layer
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential


# In[2]:


# from nltk.corpus import stopwords 


# # Model

# In[3]:


def eval_model(X_test, y_test, model):
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[4]:


"""Pass the dataset with data in the format: class:content"""
"""Text pre-processing: Removing links, special characters, and digits. df_column[1] is also converted into lower case"""
def preprocess_dataset(df, df_column):
    df[df_column[1]] = df[df_column[1]].str.replace('-', ' ')
    df[df_column[1]] = df[df_column[1]].str.replace('(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '')
    df[df_column[1]] = df[df_column[1]].str.replace('[^a-zA-Z0-9 \n]', ' ')
    df[df_column[1]] = df[df_column[1]].str.replace('\d+', '')
    df[df_column[1]] = df[df_column[1]].str.lower()
    return df


# In[5]:


"""To label the sentiment classes using integers. Not to be used for the neural network"""
def to_categorical(df):
    df.sentiment = pd.Categorical(df.sentiment)
    df['class'] = df.sentiment.cat.codes
    return df['class']


# In[6]:


"""Function returns the one-hot representation of the sentiment classes"""
def to_OneHot(df, df_columns):
    b = pd.get_dummies(df[df_column[0]], prefix="")
    list1 = list(b)
    OneHot = b[list1[0]]
    OneHot = np.column_stack(b[list1[i]] for i in range(len(list1)))
    print(len(list1))
    print(OneHot)
    return OneHot


# # Data Preprocessing

# In[8]:


df = pd.read_csv('../Datasets/Twitter Sentiment Analysis/train_data.csv')
df_column = list(df)      #Names of the columns of the dataframe
classes = df[df_column[0]].unique().size #Number of distinct classes for the dataset. 13 for the given dataset
df = preprocess_dataset(df, df_column)
df.head()


# In[9]:


labels = to_OneHot(df, df_column)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(df[df_column[1]], labels, test_size = 0.2, random_state = 10)


# # Sentence to List

# In[11]:


X_train = X_train.to_frame().reset_index()
X_test = X_test.to_frame()
# print(y_train.shape)
# X_train.shape


# In[12]:


train_sent_list = [list(filter(None, row['content'].split(" "))) for i, row in X_train.iterrows()]
test_sent_list = [list(filter(None, row['content'].split(" "))) for i, row in X_test.iterrows()]
# print(text_list[1])


# ### Removing Stopwords

# In[13]:


# text_list_reduced = []
# for i in range(len(train_sent_list)):
#     text_list_reduced.append([word for word in train_sent_list[i] if word not in stopwords.words('english')])
# train_sent_list = text_list_reduced
# del text_list_reduced


# In[14]:


print('wishes' not in train_sent_list[1])


# ### Making Vocabulary List

# In[15]:


vocab = []
for i in range(len(train_sent_list)):
    for j in range(len(train_sent_list[i])):    
        if train_sent_list[i][j] not in vocab:
            vocab.append(train_sent_list[i][j])
#     if len(text_list[i]) != 0:
#         np.asarray(X[i]/len(text_list[i]))
# X = np.asarray(X)        


# In[16]:


len(vocab)


# ### Converting sentence list to Network input

# In[17]:


"""Making sentence list using vocab indices"""
X_train_indexed  = []
for i in range(len(train_sent_list)):
    X_train_indexed.append([])
    for j in range(len(train_sent_list[i])):
        X_train_indexed[i].append(vocab.index(train_sent_list[i][j]))


# In[18]:


X_test_indexed = []
for i in range(len(test_sent_list)):
    X_test_indexed.append([])
    for j in range(len(test_sent_list[i])):
        if(test_sent_list[i][j] in vocab):
            X_test_indexed[i].append(vocab.index(test_sent_list[i][j]))


# In[19]:


X_train_reduced = keras.preprocessing.sequence.pad_sequences(X_train_indexed)
max_sent_length = len(X_train_reduced[0])
X_test_reduced= keras.preprocessing.sequence.pad_sequences(X_test_indexed, maxlen=max_sent_length)


# In[20]:


print(len(X_train_reduced[0]))


# ## Model

# In[21]:


def modelDef(vocab_size, max_sent_len, embedding_size=128,optimizer = 'adam', loss = 'categorical_crossentropy', plot = False):
    model = Sequential()
    model.add(layer.embeddings.Embedding(vocab_size, embedding_size, input_length = max_sent_len)) # max vocab index, embedding size, input list size
    model.add(layer.convolutional.Conv1D(256, 32, padding = 'same', activation = 'elu'))
#     model.add(layer.Dropout(0.2))
#     model.add(layer.pooling.MaxPooling1D(pool_size = 8))
    model.add(LSTM(128))
#     model.add(layer.Flatten())
#     model.add(Dense(128, activation='relu'))
    model.add(Dense(13, activation = 'softmax'))
    model.compile(optimizer = optimizer, loss=loss, metrics = ['accuracy'] )
    if(plot):
        plot_model(model, to_file = 'model.png')
    return model


# ## Training Network

# In[22]:


model = modelDef(vocab_size=len(vocab), max_sent_len=len(X_train_reduced[0]))


# In[ ]:


model.fit(X_train_reduced, y_train, batch_size = 1080, nb_epoch = 3, verbose = 1)


# ## Testing the Network

# In[ ]:


eval_model(X_test_reduced, y_test, model)

