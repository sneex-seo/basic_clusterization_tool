#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster
from nltk.stem.snowball import SnowballStemmer 


# In[ ]:


stemmer = PorterStemmer()
stemmer = SnowballStemmer("english") 


# In[ ]:


with open('keywords.txt', 'r+', encoding='utf-8') as f:
    keywords = [line.strip() for line in f]


# In[ ]:


def tokenizer(keyword):
    return [stemmer.stem(w) for w in keyword.split()]


# In[ ]:


tfidf = TfidfVectorizer(use_idf=False, norm=None)


# In[ ]:


pd.DataFrame(tfidf.fit_transform(keywords).toarray(), index = keywords, columns = tfidf.get_feature_names())


# In[ ]:


tfidf = TfidfVectorizer(tokenizer=tokenizer)
x = pd.DataFrame(tfidf.fit_transform(keywords).toarray(),
                 index = keywords, columns = tfidf.get_feature_names())
print(x)


# In[ ]:


c = cluster.AffinityPropagation()


# In[ ]:


c = cluster.AffinityPropagation()
c.fit_predict(x)


# In[ ]:


x['pred'] = c.fit_predict(x)


# In[ ]:


x['pred'] = c.fit_predict(x)
print(x)
x.to_csv('result.csv')

