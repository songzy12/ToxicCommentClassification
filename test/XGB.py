import csv
import errno
import glob
import os
import re
import string
import sys

from collections import OrderedDict, Counter
from subprocess import check_call
from shutil import copyfile

from tqdm import tqdm

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn import decomposition, ensemble, metrics, model_selection, naive_bayes, preprocessing, pipeline
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import StanfordNERTagger

from keras import initializers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import GlobalAveragePooling1D,Merge,Lambda,Input,GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D,TimeDistributed
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence, text
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

pd.options.mode.chained_assignment = None

train_path = '../input/train.csv'
test_path = '../input/test.csv'


## Read the train and test dataset and check the top few lines ##
# TODO:
train_df = pd.read_csv(train_path).fillna('NANN')
test_df = pd.read_csv(test_path).fillna('NANN')

train_df.shape[0]
test_df.shape[0]
train_df.head()
test_df.head()

labels = list(train_df)[2:]
len(labels)
labels
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_df[labels].head()

label0 = labels[0]
train_df[label0].head()

train_df[train_df['toxic']==1]['comment_text'].head()

## Prepare the data for modeling ###
train_y = train_df[labels].values
test_id = test_df['id'].values
labels
train_y

cols_to_drop = ['id']
train_df = train_df.drop(cols_to_drop + labels, axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)

for df in [train_df, test_df]:
    df['split'] = df['comment_text'].apply(word_tokenize)
    print('split finished...')

def feature_ner(df):
    st = StanfordNERTagger('../stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz',
                          '../stanford-ner-2017-06-09/stanford-ner.jar',
                          encoding='utf-8')
    st = Ner(host='localhost',port=9199)
    df['st'] = df['comment_text'].apply(lambda x: [w[1] for w in st.get_entities(x)])
    df['n_person'] = df['st'].apply(lambda x: x.count('PERSON'))
    df['n_location'] = df['st'].apply(lambda x: x.count('LOCATION'))
    print('NER finished...')

def feature_sid(df):
    sid = SentimentIntensityAnalyzer()
    df['sid'] = df['comment_text'].apply(sid.polarity_scores)
    for k in ['neu', 'compound', 'pos', 'neg']:
        df['sid_'+k] = df['sid'].apply(lambda x: x[k])
    print('polarity_scores finished...') 

def feature_pos(df): 
    df['pos_tag'] = df['split'].apply(lambda x: [w[1] for w in nltk.pos_tag(x)])
    for pos in ['CC', 'RB', 'IN', 'NN', 'VB', 'VBP', 'JJ', 'PRP', 'TO', 'DT']:
        df['n_pos_' + pos] = df['pos_tag'].apply(lambda x: x.count(pos))
    print('pos_tag finished...')

def feature_cnt(df):
    ## Number of words in the text ##
    df["num_words"] = df["split"].apply(len)
    
    ## Number of unique words in the text ##
    df["num_unique_words"] = df["split"].apply(lambda x: len(set(x)))
    
    ## Number of characters in the text ##
    df["num_chars"] = df['comment_text'].apply(len)
    
    ## Number of stopwords in the text ##
    #eng_stopwords = set(stopwords.words("english"))
    df["num_stopwords"] = df["split"].apply(lambda x: len([w for w in x if w in eng_stopwords]))
    
    ## Number of punctuations in the text ##
    df["num_punctuations"] = df['split'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
    
    ## Number of title case words in the text ##
    df["num_words_upper"] = df["split"].apply(lambda x: len([w for w in x if w.isupper()]))
    
    ## Number of title case words in the text ##
    df["num_words_title"] = df["split"].apply(lambda x: len([w for w in x if w.istitle()]))
    
    ## Average length of the words in the text ##
    df["mean_word_len"] = df["split"].apply(lambda x: np.mean([len(w) for w in x]))


    anchor_words = ['the', 'a', 'appear', 'little', 'was', 'one', 'two', 'three', 'ten', 'is', 
                    'are', 'ed', 'however', 'to', 'into', 'about', 'th', 'er', 'ex', 'an', 
                    'ground', 'any', 'silence', 'wall']

    gender_words = ['man', 'woman', 'he', 'she', 'her', 'him', 'male', 'female']

    for word in anchor_words + gender_words:
        df['n_'+word] = df["split"].apply(lambda x: len([w for w in x if w.lower() == word]))

for df in [train_df, test_df]:
    feature_cnt(df)

def feature_lda(train_df, test_df): 
    
    ### Fit transform the tfidf vectorizer ###
    tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
    full_tfidf = tfidf_vec.fit_transform(train_df['comment_text'].values.tolist() + test_df['comment_text'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['comment_text'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['comment_text'].values.tolist())
    
    no_topics = 20 
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(full_tfidf)
    train_lda = pd.DataFrame(lda.transform(train_tfidf))
    test_lda = pd.DataFrame(lda.transform(test_tfidf))
    
    train_lda.columns = ['lda_'+str(i) for i in range(no_topics)]
    test_lda.columns = ['lda_'+str(i) for i in range(no_topics)]
    train_df = pd.concat([train_df, train_lda], axis=1)
    test_df = pd.concat([test_df, test_lda], axis=1)
    del full_tfidf, train_tfidf, test_tfidf, train_lda, test_lda

    print("LDA finished...")


# load the GloVe vectors in a dictionary:

wv = '../input/glove.6B.100d.txt'

def loadWordVecs():
    embeddings_index = {}
    f = open(wv)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def sent2vec(embeddings_index,s): # this function creates a normalized vector for the whole sentence
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stopwords.words('english')]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(100)
    return v / np.sqrt((v ** 2).sum())

def doGlove(x_train,x_test):
    embeddings_index = loadWordVecs()
    # create sentence vectors using the above function for training and validation set
    xtrain_glove = [sent2vec(embeddings_index,x) for x in tqdm(x_train)]
    xtest_glove = [sent2vec(embeddings_index,x) for x in tqdm(x_test)]
    xtrain_glove = np.array(xtrain_glove)
    xtest_glove = np.array(xtest_glove)
    return xtrain_glove,xtest_glove,embeddings_index

glove_vecs_train,glove_vecs_test,embeddings_index = doGlove(train_df['comment_text'], test_df['comment_text'])
train_df[['sent_vec_'+str(i) for i in range(100)]] = pd.DataFrame(glove_vecs_train.tolist())
test_df[['sent_vec_'+str(i) for i in range(100)]] = pd.DataFrame(glove_vecs_test.tolist())
print("Glove sentence vector finished...")

# Using Neural Networks and Facebook's Fasttext
earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

# NN
def doAddNN(X_train,X_test,pred_train,pred_test):
    for i in range(6):
        X_train['nn_'+str(i)] = pred_train[:,i]
        X_test['nn_'+str(i)] = pred_test[:,i]
    return X_train,X_test

def initNN(nb_words_cnt,max_len):
    model = Sequential()
    model.add(Embedding(nb_words_cnt,32,input_length=max_len))
    model.add(Dropout(0.3))
    model.add(Conv1D(64,
                     5,
                     padding='valid',
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return model

def doNN(X_train,X_test,Y_train):
    max_len = 70
    nb_words = 10000
    
    print('Processing text dataset')
    texts_1 = []
    for text in X_train['comment_text']:
        texts_1.append(text)

    print('Found %s texts.' % len(texts_1))
    test_texts_1 = []
    for text in X_test['comment_text']:
        test_texts_1.append(text)
    print('Found %s texts.' % len(test_texts_1))
    
    tokenizer = Tokenizer(num_words=nb_words)
    tokenizer.fit_on_texts(texts_1 + test_texts_1)
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)

    xtrain_pad = pad_sequences(sequences_1, maxlen=max_len)
    xtest_pad = pad_sequences(test_sequences_1, maxlen=max_len)
    del test_sequences_1
    del sequences_1
    nb_words_cnt = min(nb_words, len(word_index)) + 1

    # we need to binarize the labels for the neural net
    
    # ytrain_enc = np_utils.to_categorical(Y_train)
    ytrain_enc = Y_train
        
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([xtrain_pad.shape[0], len(labels)])
    for dev_index, val_index in kf.split(xtrain_pad):
        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initNN(nb_words_cnt,max_len)
        model.fit(dev_X, y=dev_y, batch_size=32, epochs=4, verbose=1,validation_data=(val_X, val_y),callbacks=[earlyStopping])
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(xtest_pad)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
    return doAddNN(X_train,X_test,pred_train,pred_full_test/5)

train_df,test_df = doNN(train_df,test_df,train_y)
print('NN finished...')

## NN Glove

def doAddNN_glove(X_train,X_test,pred_train,pred_test):
    for i in range(6):
        X_train['nn_glove_'+str(i)] = pred_train[:,i]
        X_test['nn_glove_'+str(i)] = pred_test[:,i]
    return X_train,X_test

def initNN_glove():
    # create a simple 3 layer sequential neural net
    model = Sequential()

    model.add(Dense(128, input_dim=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(len(labels)))
    model.add(Activation('softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def doNN_glove(X_train,X_test,Y_train,xtrain_glove,xtest_glove):
    # scale the data before any neural net:
    scl = preprocessing.StandardScaler()
    #ytrain_enc = np_utils.to_categorical(Y_train)
    ytrain_enc = Y_train
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    xtrain_glove = scl.fit_transform(xtrain_glove)
    xtest_glove = scl.fit_transform(xtest_glove)
    pred_train = np.zeros([xtrain_glove.shape[0], len(labels)])
    
    for dev_index, val_index in kf.split(xtrain_glove):
        dev_X, val_X = xtrain_glove[dev_index], xtrain_glove[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initNN_glove()
        model.fit(dev_X, y=dev_y, batch_size=32, epochs=10, verbose=1,validation_data=(val_X, val_y),callbacks=[earlyStopping])
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(xtest_glove)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
    return doAddNN_glove(X_train,X_test,pred_train,pred_full_test/5)

train_df,test_df = doNN_glove(train_df,test_df,train_y,glove_vecs_train,glove_vecs_test)
print('NN Glove finished...')

# Fast Text

def doAddFastText(X_train,X_test,pred_train,pred_test):
    for i in range(6):
        X_train['ff_'+str(i)] = pred_train[:,i]
        X_test['ff_'+str(i)] = pred_test[:,i]
    return X_train,X_test


def initFastText(embedding_dims,input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(len(labels), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def preprocessFastText(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text

def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    docs = []
    for doc in df['comment_text']:
        doc = preprocessFastText(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    
    return docs

def doFastText(X_train,X_test,Y_train):
    min_count = 2

    docs = create_docs(X_train)
    tokenizer = Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(docs)
    num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

    tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
    tokenizer.fit_on_texts(docs)
    docs = tokenizer.texts_to_sequences(docs)

    maxlen = 300

    docs = pad_sequences(sequences=docs, maxlen=maxlen)
    input_dim = np.max(docs) + 1
    embedding_dims = 20

    # we need to binarize the labels for the neural net
    #ytrain_enc = np_utils.to_categorical(Y_train)
    ytrain_enc = Y_train

    docs_test = create_docs(X_test)
    docs_test = tokenizer.texts_to_sequences(docs_test)
    docs_test = pad_sequences(sequences=docs_test, maxlen=maxlen)
    xtrain_pad = docs
    xtest_pad = docs_test
    
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([xtrain_pad.shape[0], len(labels)])
    for dev_index, val_index in kf.split(xtrain_pad):
        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initFastText(embedding_dims,input_dim)
        model.fit(dev_X, y=dev_y, batch_size=32, epochs=25, verbose=1,validation_data=(val_X, val_y),callbacks=[earlyStopping])
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(docs_test)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
    return doAddFastText(X_train,X_test,pred_train,pred_full_test/5)

train_df,test_df = doFastText(train_df,test_df,train_y)
print('FastText finished...')


cols_to_drop = ['comment_text', 'split']
train_X = train_df.drop(cols_to_drop, axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

### Fit transform the tfidf vectorizer ###
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
full_tfidf = tfidf_vec.fit_transform(train_df['comment_text'].values.tolist() + test_df['comment_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['comment_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['comment_text'].values.tolist())

def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)

    return pred_test_y, pred_test_y2, model

cv_scores = []
pred_full_test = np.zeros([test_df.shape[0], len(labels)])
pred_train = np.zeros([train_df.shape[0], len(labels)])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    dev_X, dev_y
    for i, j in enumerate(labels):
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y[:,i], val_X, val_y[:,i], test_tfidf)
        pred_test_y[:, 0]
        pred_full_test[:, i] = pred_full_test[:,i] + pred_test_y[:,0] # FIXME
        pred_train[val_index,i] = pred_val_y[:,0]
        cv_scores.append(metrics.log_loss(val_y[:,i], pred_val_y[:,0]))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

### Fit transform the count vectorizer ###
tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit(train_df['comment_text'].values.tolist() + test_df['comment_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['comment_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['comment_text'].values.tolist())

cv_scores = []
pred_full_test = np.zeros([test_df.shape[0], len(labels)])
pred_train = np.zeros([train_df.shape[0], len(labels)])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    dev_X, dev_y
    for i, j in enumerate(labels):
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y[:,i], val_X, val_y[:,i], test_tfidf)
        pred_test_y[:, 0]
        pred_full_test[:, i] = pred_full_test[:,i] + pred_test_y[:,0] # FIXME
        pred_train[val_index,i] = pred_val_y[:,0]
        cv_scores.append(metrics.log_loss(val_y[:,i], pred_val_y[:,0]))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.


# add the predictions as new features #
for i in range(6):
    train_df["nb_cvec_"+str(i)] = pred_train[:,i]
    test_df["nb_cvec_"+str(i)] = pred_full_test[:,i]
print("Naive Bayesian Count Vector finished...")

### Fit transform the tfidf vectorizer ###
tfidf_vec = CountVectorizer(ngram_range=(1,7), analyzer='char')
tfidf_vec.fit(train_df['comment_text'].values.tolist() + test_df['comment_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['comment_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['comment_text'].values.tolist())

cv_scores = []
pred_full_test = np.zeros([test_df.shape[0], len(labels)])
pred_train = np.zeros([train_df.shape[0], len(labels)])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    dev_X, dev_y
    for i, j in enumerate(labels):
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y[:,i], val_X, val_y[:,i], test_tfidf)
        pred_test_y[:, 0]
        pred_full_test[:, i] = pred_full_test[:,i] + pred_test_y[:,0] # FIXME
        pred_train[val_index,i] = pred_val_y[:,0]
        cv_scores.append(metrics.log_loss(val_y[:,i], pred_val_y[:,0]))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.


# add the predictions as new features #
for i in range(6):
    train_df["nb_cvec_char_"+str(i)] = pred_train[:,i]
    test_df["nb_cvec_char_"+str(i)] = pred_full_test[:,i]
print("Naive Bayersian Count Vector Char finished...")

### Fit transform the tfidf vectorizer ###
tfidf_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='char')
full_tfidf = tfidf_vec.fit_transform(train_df['comment_text'].values.tolist() + test_df['comment_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['comment_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['comment_text'].values.tolist())

cv_scores = []
pred_full_test = np.zeros([test_df.shape[0], len(labels)])
pred_train = np.zeros([train_df.shape[0], len(labels)])
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    dev_X, dev_y
    for i, j in enumerate(labels):
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y[:,i], val_X, val_y[:,i], test_tfidf)
        pred_test_y[:, 0]
        pred_full_test[:, i] = pred_full_test[:,i] + pred_test_y[:,0] # FIXME
        pred_train[val_index,i] = pred_val_y[:,0]
        cv_scores.append(metrics.log_loss(val_y[:,i], pred_val_y[:,0]))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

# add the predictions as new features #
for i in range(6):
    train_df["nb_tfidf_char_"+str(i)] = pred_train[:,i]
    test_df["nb_tfidf_char_"+str(i)] = pred_full_test[:,i]
print("Naive Bayersian TFIDF Vector Char finished...")

n_comp = 20
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd

def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = len(labels)
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_df.shape[0], len(labels)])
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index,:] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("cv scores : ", cv_scores)

pred_full_test /= 5.0

out_df = pd.DataFrame(pred_full_test)
out_df.columns = labels
out_df.insert(0, 'id', test_id)
out_df.to_csv("../output/result.csv", index=False)
