
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from gensim.models import KeyedVectors

from nltk.corpus import stopwords

import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
import keras.layers as layer
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential


# In[2]:


def eval_model(X_test, y_test, model):
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[3]:


def preprocess_dataset(df, df_column):
    df[df_column[1]] = df[df_column[1]].str.replace('-', ' ')
    df[df_column[1]] = df[df_column[1]].str.replace('(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '')
    df[df_column[1]] = df[df_column[1]].str.replace('[^a-zA-Z0-9 \n]', ' ')
    df[df_column[1]] = df[df_column[1]].str.replace('\d+', '')
    df[df_column[1]] = df[df_column[1]].str.lower()
    return df


# In[4]:


# def to_categorical(df):
#     df.sentiment = pd.Categorical(df.sentiment)
#     df['class'] = df.sentiment.cat.codes
#     return df['class']


# In[5]:


def to_OneHot(df, df_columns):
    b = pd.get_dummies(df[df_column[0]], prefix="")
    list1 = list(b)
    OneHot = b[list1[0]]
    OneHot = np.column_stack(b[list1[i]] for i in range(len(list1)))
    print(len(list1))
    print(OneHot)
    return OneHot


# In[10]:


def get_sent_word_list(a):
    b = [list(filter(None, row['content'].split(" "))) for i, row in a.iterrows()]
    return b


# In[11]:


def remove_stopwords(a):
    text_list_reduced = []
    for i in range(len(a)):
        text_list_reduced.append([word for word in a[i] if word not in stopwords.words('english')])
    return text_list_reduced


# In[12]:


def word_list_to_sent(a):
    a = [" ".join(a[i]) for i in range(len(a))]
    return a


# In[15]:


def get_vocab(train_sent_list, min_count):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_list_to_sent(train_sent_list))
    d = dict(tokenizer.word_counts)
    filtered_dict = {k:v for k,v in d.items() if v>=min_count}
    vocab = list(filtered_dict.keys())
    vocab.append("UNK")
    return vocab


# In[18]:


def add_unk(sent_list, vocab):
    for i in range(len(sent_list)):
        for j in range(len(sent_list[i])):
            if sent_list[i][j] not in vocab:
                sent_list[i][j]= "UNK"
    return sent_list


# In[ ]:


def get_embedding_index(model):
    embedding_index = {}
    for word in model.wv.vocab.keys():
        embedding_index[word] = model.wv[word]
    return embedding_index


# In[ ]:


def word_to_index(sent_list, vocab):
    indexed = []    
    for i in range(len(sent_list)):
        indexed_sen = []
        for j in range(len(sent_list[i])):
            indexed_sen.append(vocab.index(sent_list[i][j]))
        indexed.append(indexed_sen)
    return indexed


# In[ ]:


def get_embedding_matrix(vocab, embedding_index, embed_vec_len = 300):
    embedding_matrix = np.zeros((len(vocab), embed_vec_len))
    for i in range(len(vocab)):
        if vocab[i] in model.wv.vocab:
            embedding_matrix[i] = embedding_index[vocab[i]]
            print(i) #check if the function is working
    return embedding_matrix


# In[ ]:


def modelDef(vocab_size, embedding_matrix, max_sent_len, embedding_size=300,optimizer = 'adam', loss = 'categorical_crossentropy', plot = False):
    model = Sequential()
    model.add(layer.embeddings.Embedding(vocab_size, embedding_size, weights = [embedding_matrix], input_length = max_sent_len, trainable=False)) # max vocab index, embedding size, input list size
#     model.add(layer.convolutional.Conv1D(256, 32, padding = 'same', activation = 'elu'))
#     model.add(layer.Dropout(0.20))
#     model.add(layer.pooling.MaxPooling1D(pool_size = 8))
#     model.add(layer.Flatten())
    model.add(LSTM(100))
    model.add(layer.Dropout(0.20))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(13, activation = 'softmax'))
    model.compile(optimizer = optimizer, loss=loss, metrics = ['accuracy'] )
    if(plot):
        plot_model(model, to_file = 'model.png')
    return model


# In[6]:


df = pd.read_csv('./Datasets/Twitter Sentiment Analysis/train_data.csv')
df_column = list(df)      #Names of the columns of the dataframe
classes = df[df_column[0]].unique().size #Number of distinct classes for the dataset. 13 for the given dataset
df = preprocess_dataset(df, df_column)
df.head()


# In[7]:


df[df_column[1]].tolist()


# In[8]:


labels = to_OneHot(df, df_column)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(df[df_column[1]].to_frame(), labels, test_size = 0.2, random_state = 10)


# In[13]:


train_sent_list = get_sent_word_list(X_train)
test_sent_list = get_sent_word_list(X_test)


# In[14]:


train_sent_list = remove_stopwords(train_sent_list)
test_sent_list = remove_stopwords(test_sent_list)


# In[16]:


min_count = 3
vocab = get_vocab(train_sent_list, min_count)


# In[17]:


len(vocab)


# In[19]:


train_sent_list = add_unk(train_sent_list, vocab)
test_sent_list = add_unk(test_sent_list, vocab)


# In[20]:


model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True)


# In[ ]:


embedding_index = get_embedding_index(model)


# In[ ]:


X_train_indexed = word_to_index(train_sent_list, vocab)


# In[ ]:


X_test_indexed = word_to_index(test_sent_list, vocab)


# In[ ]:


embedding_matrix =get_embedding_matrix(vocab, embedding_index)


# In[ ]:


X_train_reduced = keras.preprocessing.sequence.pad_sequences(X_train_indexed)
max_sent_length = len(X_train_reduced[0])
X_test_reduced= keras.preprocessing.sequence.pad_sequences(X_test_indexed, maxlen=max_sent_length)


# In[ ]:


print(len(X_train_reduced[0]))


# In[ ]:


model = modelDef(len(vocab), embedding_matrix, max_sent_length)


# In[ ]:


model.fit(X_train_reduced, y_train, batch_size = 384, nb_epoch = 3, verbose = 1)


# In[ ]:


eval_model(X_test_reduced, y_test, model)

