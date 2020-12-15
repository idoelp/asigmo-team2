#!/usr/bin/env python
# coding: utf-8

# #### First things first

# In[91]:


import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error


# In[92]:


df = pd.read_csv('C:/Users/User/Downloads/train_humor.csv').drop(columns = 'id')
df.head()


# #### selecting rows

# In[93]:


X = df[['text']] #4932 Do not reset the index!
y = df[['offense_rating']] #4932


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)
"X_train and y_train shape: {0}, {1}, X_test and y_test shape {2}, {3}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[16]:


y_train.describe()


# In[94]:


_, _, histogram = plt.hist(y_train['offense_rating'], bins = 1000, histtype = 'step')
histogram


# #### helper functions

# In[95]:


def stemmer(text, stemmer):
    return(' '.join([stemmer.stem(w) for w in word_tokenize(text)]))

def count_words(input):
    """ Returns number of occurences of characters specified in char """     
    return len(input.split())

def remove_punctuation(s_input, include_char = None):
    """ Returns input string without punctuation """
    import string as String
    punct = String.punctuation
    
    if not include_char is None:
        index = String.punctuation.index(include_char)
        punct = String.punctuation[:index] + String.punctuation[(index + 1):]
        
    punct += '\n'
        
    translator = str.maketrans(punct, ' '*len(punct))
    
    return s_input.translate(translator)

def remove_stopwords(text, use_stopwords = None, df = True, exclude_number = True):
    """ Returns input string removing stopwords from it. """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    if use_stopwords is None:
        use_stopwords = set(stopwords.words("english"))
        
    if df:
        new_text = word_tokenize(text)
        if exclude_number:
            new_text = [word for word in new_text if not word.isnumeric()]
        new_text = " ".join([word for word in new_text if word not in use_stopwords])
    else:
        new_text = ""
        for word in text:
            if word not in use_stopwords:
                new_text += word + " "

    return new_text

def sep_upper(text):
    """ Take a text as input and insert space before every uppercase letter. """
    
    new_text = ""
    for letter in text:
        if letter.isupper():
            new_text += " " + letter
        else:
            new_text += letter
    
    return new_text

def remove_space(text):
    return(re.sub(' +',' ',text)) 


# #### (basic) pre-process of text columns

# In[96]:


def pre_proc(text_col):
    text_col = text_col.apply(remove_punctuation) # removes String.punctuation characters
    #text_col = text_col.apply(remove_stopwords)   # removes english stopwords 
    text_col = text_col.str.replace('[^\w\s]','').str.strip() # and removes whitespaces
    text_col = text_col.apply(sep_upper) # adds space before an uppercase
    text_col = text_col.str.lower() # lowercase
    
    return text_col


# In[97]:


X_train.text = pre_proc(X_train.text)
X_test.text = pre_proc(X_test.text)


# #### basic new features

# In[71]:


X_train['qtd_words'] = X_train.text.apply(count_words)
X_test['qtd_words'] = X_test.text.apply(count_words)


# #### wait for it

# In[72]:


vectorizer = CountVectorizer()


X_train_trans = pd.DataFrame(vectorizer.fit_transform(X_train.text).toarray()
                             , columns = vectorizer.get_feature_names()
                             , index = X_train.index)
X_train_trans['qtd_words'] = X_train['qtd_words']

X_test_trans = pd.DataFrame(vectorizer.transform(X_test.text).toarray()
                            , columns = vectorizer.get_feature_names()
                            , index = X_test.index)
X_test_trans['qtd_words'] = X_test['qtd_words']
#X_test.text = vectorizer.transform(X_test.text).toarray()
# print(vectorizer.get_feature_names())


# In[103]:


X_train.head()


# In[99]:


get_ipython().system('pip install xgboost')


# In[104]:


import xgboost as xgb


# In[105]:


xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)

xgb_model.fit(X_train_trans, y_train)

y_pred_xgb = xgb_model.predict(X_test_trans)

mse=mean_squared_error( y_test, y_pred_xgb)

print(np.sqrt(mse))


# In[ ]:




