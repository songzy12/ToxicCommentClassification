import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

import tensorflow as tf

callbacks.EarlyStopping(monitor='val_loss',
                        min_delta=0,
                        patience=2,
                        verbose=0, mode='auto')

path = '../input/'
comp = './'
EMBEDDING_FILE=path+'glove.6B.100d.txt'
TRAIN_DATA_FILE=path+'train.csv'
TEST_DATA_FILE=path+'test.csv'

# NOTE: https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge
# Here we can get a estimation of our hyper-parameters
# TODO: tune this embed size 
embed_size = 100 # how big is each word vector
# wordCount = {'word2vec':66078,'glove':81610,'fasttext':59613,'baseline':210337}
max_features = 80000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 200 # max number of words in a comment to use

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

print('len(train):', len(train), 'len(test):', len(test))

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
print('emb_mean:',emb_mean,'emb_std:',emb_std)

# NOTE: dictionary mapping words (str) to their rank/index (int)
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
print('max_features:',max_features,'len(word_index):',len(word_index))
# NOTE: here is how we can randomly initialize unknown word embeddings
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value    

inp = Input(shape=(maxlen,))
# NOTE: trainable=False
x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=False)(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(len(list_classes), activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc_roc])

# TODO: delete validation_split for submission
# batch_size: len(train): 159571 len(test): 153164
# epochs: check the valid loss and training loss

history = model.fit(X_t, y, batch_size=1024, epochs=16, validation_split=0.1)

# print(history.history.keys())
# print(history.history['val_loss'])

y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(path+'sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('../output/lstm_glove_submission.csv', index=False)
