from util import *
pd.options.mode.chained_assignment = None

## Read the train and test dataset and check the top few lines ##
# TODO:

src_path = '../'
input_path = src_path+'../input/'
train_df = pd.read_csv(input_path+'train.csv').fillna('NANN')
test_df = pd.read_csv(input_path+'test.csv').fillna('NANN')

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
    eng_stopwords = set(stopwords.words("english"))
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

#df.to_pickle(file_name)
#df = pd.read_pickle(file_name)
try:
    train_df.to_pickle(src_path+'../output/train_df.pkl')
    test_df.to_pickle(src_path+'../output/test_df.pkl')
except:
    print(sys.exc_info()[0])
    code.interact(local=locals())

cols_to_drop = ['comment_text', 'split']
train_X = train_df.drop(cols_to_drop, axis=1)
test_X = test_df.drop(cols_to_drop, axis=1)

### Fit transform the tfidf vectorizer ###
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
full_tfidf = tfidf_vec.fit_transform(train_df['comment_text'].values.tolist() + test_df['comment_text'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['comment_text'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['comment_text'].values.tolist())

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

def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)

    return pred_test_y, pred_test_y2, model

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

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=400):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.12
    param['max_depth'] = 5
    param['silent'] = 1
    param['eval_metric'] = 'logloss'
    param['min_child_weight'] = 1
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return model, pred_test_y

preds = np.zeros((test_X.shape[0], len(labels)))

for i, j in enumerate(labels):
    print('fit '+j)
    model, pred_y = runXGB(train_X, train_y[:,i], test_X)
    preds[:,i] = pred_y

out_df = pd.DataFrame(preds)
out_df.columns = labels
out_df.insert(0, 'id', test_id)
code.interact(local=locals())
out_df.to_csv(src_path+"../output/submission_xgb.csv", index=False)
