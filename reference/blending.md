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
