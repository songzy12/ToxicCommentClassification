import numpy as np
import pandas as pd
import os
import code

output_path = '../output/'
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

cnt = 0
p0 = None
for f in os.listdir(output_path):
    cnt += 1
    p = pd.read_csv(output_path+f)
    if p0 is None:
        p0 = p.copy()
    else:
        p0[label_cols] += p[label_cols]
p0[label_cols] = p0[label_cols] / cnt

p0.to_csv('submission.7z', index=False, compression='gzip')
