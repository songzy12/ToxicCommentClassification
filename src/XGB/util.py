import code
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
