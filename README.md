## TODO

* swear words

## FE

Swear words: https://github.com/Donskov7/toxic_comments/tree/master/data

## Ensemble

* Average of LSTM + Glove and TFIDF (char/word) + SVM and BiLSTM + Attention: 

  **score: 0.041, rank: 169**


* Average of LSTM + GloVe and TFIDF (char/word) + SVM: 

  **score: 0.042, rank: 139**

* Average of LSTM and NB+SVM: 

  **score: 0.052, rank: 608**

## GloVe LSTM

```
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
```



## LSTM

```
def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
```

## NB+SVM

Just n-gram TFIDF (char & word) and Logistic Regression.