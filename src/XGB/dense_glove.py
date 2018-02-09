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
    model.add(Activation('sigmoid'))

    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam')
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


