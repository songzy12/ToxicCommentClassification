## stacking

whatever.

## blending

<https://www.linkedin.com/pulse/do-you-want-learn-stacking-blending-ensembling-machine-soledad-galli/>

* https://www.kaggle.com/prashantkikani/hight-of-blend-v2:
* 0.2 * lgb-gru-lr-lstm-nb-svm-average-ensemble + 
  * https://www.kaggle.com/peterhurford/lgb-gru-lr-lstm-nb-svm-average-ensemble
  * 0.4 * pooled-gru-fasttext + 0.3 * minimal-lstm-nb-svm-baseline-ensemble + 0.15 * lightgbm-with-select-k-best-on-tfidf + 0.15 * logistic-regression-with-words-and-char-n-grams
* 0.6 * bi-gru-cnn-poolongs + 
  * https://www.kaggle.com/konohayui/bi-gru-cnn-poolings
* 0.2 * blend-of-blends-1
  * https://www.kaggle.com/tunguz/blend-of-blends-1
  * 0.5 * lgb-gru-lr-lstm-nb-svm-average-ensemble + 0.5 * toxic-one-more-b8bce2

```
# All credits goes to original authors.. Just another blend...
import pandas as pd
from sklearn.preprocessing import minmax_scale
sup = pd.read_csv('../input/blend-of-blends-1/superblend_1.csv')
allave = pd.read_csv('../input/lgb-gru-lr-lstm-nb-svm-average-ensemble/submission.csv')
gru = pd.read_csv('../input/bi-gru-cnn-poolings/submission.csv')

blend = allave.copy()
col = blend.columns

col = col.tolist()
col.remove('id')
# keeping weight of single best model higher than other blends..
blend[col] = 0.2*minmax_scale(allave[col].values)+0.6*minmax_scale(gru[col].values)+0.2*minmax_scale(sup[col].values)
print('stay tight kaggler')
blend.to_csv("hight_of_blend_v2.csv", index=False)
```



## TODO

* swear words

## Feature Engineer

Swear words: https://github.com/Donskov7/toxic_comments/tree/master/data

## FastText

CNN_Random + GRU_GloVe + TFIDF_LR + Attention + BiLSTM_Random + FastText

**AUC: 0.9820**

## CNN Random  

CNN_Random + GRU_GloVe + TFIDF_LR + Attention + BiLSTM_Random 

**AUC: 0.9832**

## Dense GloVe  

Dense_GloVe + GRU_GloVe + TFIDF_LR + Attention + BiLSTM_Random

**AUC: 0.9830**

## GRU GloVe 

```
def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
```

**AUC: 0.9819, rank: 1085**

 GRU_GloVe + TFIDF_LR + Attention + BiLSTM_Random

**AUC: 0.9837**

## LSTM GloVe 

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

**AUC: 0.9792, rank: 638**

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

## Attention

**AUC: 0.9754**

##Ensemble

* Bagging

* Boosting

* Voting

* Average: 

  For each single model here, the scores are:

  * Attention: 


  * LSTM + Glove: 
  * TFIDF (char/word) + SVM: 


## XGBoost

## NB+SVM

Just n-gram TFIDF (char & word) and Logistic Regression.

## Word Embedding

* GloVe
* Word2Vec
* FastText