import numpy as np, pandas as pd

f_lstm = '../output/lstm_glove_submission.csv'
f_nbsvm = '../output/tfidf_lr_submission.csv'
f_attention = '../output/attention_0.0510_simple_lstm_glove_vectors_0.25_0.25.csv'

p_lstm = pd.read_csv(f_lstm)
p_nbsvm = pd.read_csv(f_nbsvm)
p_attention = pd.read_csv(f_attention)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_lstm.copy()
p_res[label_cols] = (p_nbsvm[label_cols] + p_lstm[label_cols] + p_attention[label_cols]) / 3

p_res.to_csv('../output/submission.csv', index=False)
