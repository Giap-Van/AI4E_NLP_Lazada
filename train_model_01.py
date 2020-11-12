import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
from sklearn.externals import joblib

warnings.filterwarnings('ignore')

df = pd.read_csv('data_crawler.csv', delimiter='\t', header=None)

batch_1 = df
print(df)

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
# joblib.dump(model, 'tfidf.pkl')
tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# print(len(batch_1[0]))
labels = np.zeros(len(batch_1[0]))
for i in range(1,len(batch_1[0])):
  labels[i]=batch_1[0][i][-1]
  
print(labels.shape)

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

np.array(padded).shape

attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()

train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

joblib.dump(lr_clf, 'saved_model.pkl')
sc=lr_clf.score(test_features, test_labels)
print("score: ", sc )