https://www.kaggle.com/jagangupta/lessons-from-toxic-blending-is-the-new-sexy

## rank 1

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557

**Diverse pre-trained embeddings (baseline public LB of 0.9877)** 

For the latter, our work-horse was two BiGru layers feeding into two final Dense layers.

For the former, we searched the net for available pre-trained word embeddings and settled primarily on the highest-dimensional FastText and Glove embeddings pre-trained against Common Crawl, Wikipedia, and Twitter.

**Translations as train/test-time augmentation (TTA) (boosted LB from 0.9877 to 0.9880)** 

For the predictions, we simply averaged the predicted probabilities of the 4 comments (EN, DE, FR, ES).

This had a dramatic impact on the performance of our models. For example,

- Vanilla Bi-GRU model: 0.9862LB
- “ (w/ train-time augments): 0.9867 LB
- “ (w/ test-time augments): 0.9865 LB
- “ (w/ both train/test-time augments): 0.9874 LB

**Rough-bore pseudo-labelling (PL) (boosted LB from 0.9880 to 0.9885)**

**Robust CV + stacking framework (boosted LB from 0.9885 to 0.9890)**

For stacking, we used primarily LightGBM, which both was faster than XGBoost and reached slightly better CV scores with heavy bayesian optimization.

Parameters were selected by choosing the best out of 250 runs with bayesian optimization; key points in the parameters were small trees with low depth and strong l1 regularization. We bagged 6 runs of both DART and GBDT using different seeds to account for potential variance during stacking.

## rank 2 

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52612

For this competition we built an ensemble of RNN,DPCNN and GBM models in order to achieve the appropriate diversity.

- train on pre-trained embeddings. (FastText, Glove twitter, BPEmb, Word2Vec, LexVec)
- train and test-time augmentation (TTA) using translations to German, French and Spanish and back to English thanks to Pavel Ostyakov’s open contribution.
- train on translations to the above languages and using DE, FR, ES BPEmb pre-trained embeddings.

**Ensemble**: Finally we ended up with about 30 different models of which we took the average.

## rank 3

**Architecture**

* **First layer:** concatenated fasttext and glove twitter embeddings. Fasttext vector is used by itself if there is no glove vector but not the other way around. Words without word vectors are replaced with a word vector for a word "something". Also, I added additional value that was set to 1 if a word was written in all capital letters and 0 otherwise.
* **Second layer:** SpatialDropout1D(0.5)
* **Third layer:** Bidirectional CuDNNLSTM with a kernel size 40. I found out that LSTM as a first layer works better than GRU.
* **Fourth layer:** Bidirectional CuDNNGRU with a kernel size 40.
* **Fifth layer:** A concatenation of the last state, maximum pool, average pool and two features: "Unique words rate" and "Rate of all-caps words"
* **Sixth layer:** output dense layer.

**Hyper-parameters and preprocessing:**

- Batch size: 512. I found that bigger batch size makes results more stable.
- Epochs: 15.
- Sequence length: 900.
- Optimizer: Adam with clipped gradient.
- Preprocessing: Unidecode library (<https://pypi.python.org/pypi/Unidecode>) to convert text to ASCII first and after that filtering everything except letters and some punctuation.

**Text normalization**

I did a lot of work on fixing misspellings and I think it improved the score. I was only fixing misspellings that didn't have a fasttext vector. Things that I did:

- Created a list of words that appear more often in toxic comments than in regular comments and words that appear more often in non-toxic comments. For every misspelled word I looked up if it has a word in the list with a small Levenshtein distance to it.
- Fixed some misspellings with TextBlob dictionary.
- Fixed misspellings by finding word vector neighborhoods. Fasttext tool can create vectors for out-of-dictionary words which is really nice. I trained my own fasttext vectors on Wikipedia comments corpus and used them to do this. I also used those vectors as embeddings but results were not as good as with regular fasttext vectors.

## rank 5

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52630

The best linear model was around 0.9814 at LB and It was an LSVC (linear support vector machine) model.

Best lightgbm was around 0.9830 and had up to 2-gram words, stemming and up to 6 char char-grams along with some features generated from word2vec and pre-trained embeddings.

Our best NN was a 2-level bidirectional gru followed by max pooling and 2 fully-connected layers. Core improvements over baseline were exponential learning rate annealing, skip-connections around hidden layers, and **900d embeddings that consisted of concatenated glove, fasttext with subword information and fasttext crawl**. This scored 0.9865 (and 0.9861 private).

Other notable mentions include a char-level DPCNN and RNN trained over wordparts produced by byte-pair encoding. Other strong NNs were based on the implementation shared by Pavel Ostyakov

Also the lstm from Neptune with an additional input for chars (dual input) and stemming had a score near 0.9860.

## rank 7

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52691

The regular methods are used like clean unknown words, punctuation remove, abbreviation recover.

Translate and re-translate the train/test data using google api, then we can augment the data by #language.

- rnn models with different network structure (the similarity is that the rnn layer is always on top of embedding layer, the differences are how to concat each layer, if the attention is used, if conv layer is added extract high-level feature, etc.). GRU and LSTM performs nearly the same.
- Word level Deep Pyramid CNN(<http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf>) Word level text CNN (<https://arxiv.org/pdf/1408.5882.pdf>) Char level CNN model (<http://www.aclweb.org/anthology/E17-1104>).
- Capsule Network
- LR using tfidf char and word features
- Lightgbm/xgboost using dimensionality reduction of tfidf char and word features

Each model was trained using different languages and different fixed pre-trained word embeddings (glove, w2v, fastest in which Glove performs the best). 

The hyper-params are really important, the experience we learned is that dropout > max sentence length = max #features > #nodes in each layer.

RNN models dominated in all models!

A glimpse on the private LB showed that simple average performs the best.

## rank 11

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52526

Let me just say that on my side I had basically two models. 

First model is word embeddings, followed by bidirectional gru with return sequence true, followed by k max pooling, then dense logistic. 

Second model is the same with lstm instead of gru. Very simple, fast to train. Trained on 10 stratified folds.

```
def get_model(num_filters, top_k):

    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, 2 * num_filters * top_k))

    inp = Input(shape=(maxlen, ))
    layer = Embedding(max_features, embed_size, weights=[embedding_matrix])
    x = layer(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(num_filters, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    k_max = Lambda(_top_k)(x)
    conc = concatenate([avg_pool, k_max])
    outp = Dense(6, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
```

The only way to get reasonably good results with only 2 models was to train it with different dataset and different word embeddings. I used 7 different word embeddings and 4 datasets:

- raw data
- raw data with foreign messages translated into english
- train-es shared by Pavel (thanks a lot Pavel!)
- train-preprocessed shared by Zafar (thanks a lot Zafar!)

The word embeddings are all listed on the external data thread. Besides the usual glove, fasttext and word2vec, the last two ones were new to me:

- crawl-300d-2M.vec
- glove.840B.300d.w2vec.txt
- wiki.en.vec
- glove.twitter.27B.200d.txt
- GoogleNews-vectors-negative300.bin
- numberbatch-en.txt
- lexvec commoncrawl

The original word2vec embeddings (Google News) was the worst one, by a significant margin.

## rank 15

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52563

BPE (Byte Pair Encoding): https://github.com/bheinzerling/bpemb

Spell Correction using Embedding: https://www.kaggle.com/cpmpml/spell-checker-using-word2vec

Blending: 

* https://www.kaggle.com/hhstrand/hillclimb-ensembling/code
* http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

## rank 25

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52647

## rank 33

**Models (best private score shown):**

- CapsuleNet (*0.9860 private*, 0.9859 public)
- RNN Version 1 (*0.9858 private*, 0.9863 public)
- RNN Version 2 (*0.9856 private*, 0.9861 public)
- Two Layer CNN (*0.9826 private*, 0.9835 public)
- NB-SVM (*0.9813 private*, 0.9813 public)

**Ensembling (best private score shown):**

- Level 1a: Average 10 out-of-fold predictions (as high as *0.9860 private*, 0.9859 public)
- Level 1b: Average models with different embeddings (as high as *0.9866 private*, 0.9871 public)
- Level 2a: LightGBM Stacking (*0.9870 private*, 0.9874 public)
- Level 2b: Average multiple seeds (*0.9872 private*, 0.9876 public)

## rank 34

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52645

blending: https://www.kaggle.com/tilii7/cross-validation-weighted-linear-blending-errors

* It seems like taking almost every single public kernel, training them with 5CV, combining them with a little bit of special secret sauce and a good level 2 stacking scheme, is enough to do well at Kaggle.

- Variety in embeddings is important for model diversity.
- Varying input data is just as important, if not more important, than varying the model being used.
- Pseduo-labeling and train/test augmentation can work.
- Models that are weak on level 1 can be the best contributors to level 2, and models that are weak on level 2 can still be the best contributors to level 3.
- DART mode on LightGBM can be really useful.

## MISC

Google CoLaboratory: 

I experimented with [byte pair encoding](https://github.com/rsennrich/subword-nmt) which allows you to reduce the number of out of vocabulary words by splitting rare words into constituent elements. 

Fasttext embeddings contain subword elements, and so you can get pre-trained embeddings for almost every token in the dataset.

My experience was that the smaller vocab size didn't work very well

Padding all sequences to the same length appears to have a significant regularizing effect

I experimented with various pooling options (average, max, attention). The best turned out to be top-k pooling.