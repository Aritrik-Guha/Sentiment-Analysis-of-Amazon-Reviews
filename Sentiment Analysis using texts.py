#!/usr/bin/env python
# coding: utf-8

# In[2]:


import turicreate as tc


# # Importing of Dataset and Reading its contents

# In[3]:


sf=tc.SFrame("amazon_baby.sframe")


# In[4]:


sf


# # Visualization of dataset contents

# In[4]:


sf['rating'].show()


# In[5]:


tc.show(sf['name'],sf['rating'])


# In[6]:


sf.explore()


# In[7]:


tc.visualization.columnwise_summary(sf)


# ## Building the word count vector for each review

# In[5]:


sf['word_count']=tc.text_analytics.count_words(sf['review'])


# In[6]:


sf.explore(title='Amazon reviews')


# ### Finding the most reviewed product

# In[10]:


tc.visualization.item_frequency(sf['name'])


# #### Hereby, we find that the most reviewed product is - Vulli Sophie the Giraffe Teether

# ## Analyzing the maximum viewed product

# In[7]:


top_prod=sf[sf['name']=='Vulli Sophie the Giraffe Teether']


# In[8]:


len(top_prod)


# In[9]:


top_prod


# In[14]:


tc.visualization.set_target('auto')
top_prod['rating'].show()


# In[15]:


sf['rating'].show()


# ##### Here we can see that more people have liked amazon products, by judjing the count of amazon ratings

# ### Defining positive and negative sentiment

# In[11]:


sf['sentiment']=sf['rating']>=3


# In[12]:


sf[sf['rating']<3]


# ## Model Building

# In[13]:


train_data, test_data= sf.random_split(0.8,seed=0)


# In[14]:


sentiment_model=tc.logistic_classifier.create(train_data,features=['word_count'],target='sentiment',validation_set=test_data)


# ### Evaluation of model

# In[15]:


ev=sentiment_model.evaluate(test_data)
ev


# In[21]:


tc.evaluation.roc_curve(targets=test_data['sentiment'], predictions=sentiment_model.predict(test_data), average=None, index_map=None)


# ## Prediction/Classification of test sets

# In[22]:


sentiment_model.classify(test_data)


# In[23]:


sentiment_model.summary()


# In[24]:


sentiment_model.predict(test_data)


# In[16]:


a1=tc.SArray(test_data['sentiment'])


# In[17]:


a2=tc.SArray(sentiment_model.predict(test_data))


# In[18]:


data = {'predicted value':a2, 'real value':a1} 


# In[19]:


import pandas as pd
z=tc.SFrame(pd.DataFrame(data))


# In[20]:


z


# ### Plot evaluation/ROC curve

# In[21]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(ev['roc_curve']['fpr'], ev['roc_curve']['tpr'])


# ### Comparing predictions for top reviewed product

# In[22]:


top_prod['predicted_sentiment']=sentiment_model.predict(top_prod,output_type='probability')
top_prod


# ## Sort reviews based on predicted sentiment

# In[23]:


top_prod=top_prod.sort('predicted_sentiment',ascending=False)


# In[24]:


top_prod.explore()


# In[25]:


top_prod.tail(1)


# ## Select few words, and count their appearnance in reviews

# In[26]:


selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[27]:


sf.explore()


# In[28]:


def w_count(dict,i):
        if i in dict:
            return dict[i]
        else:
            return 0   


# In[29]:


for i in selected_words:
    sf[i] = sf['word_count'].apply(lambda x:w_count(x,i))


# In[39]:


s=[]
for i in selected_words:
    s.append(sf[i].sum())


# In[41]:


data2 = {'words':selected_words, 'appearance in reviews':s} 
z2=tc.SFrame(pd.DataFrame(data2))


# In[45]:


z2[z2['appearance in reviews']==z2['appearance in reviews'].max()]


# In[46]:


z2[z2['appearance in reviews']==z2['appearance in reviews'].min()]


# ## Sentiment Analysis using selected words

# In[48]:


train_set,test_set = sf.random_split(.8, seed=0)


# In[50]:


sentiment_model2=tc.logistic_classifier.create(train_set,features=selected_words,target='sentiment',validation_set=test_set)


#  ## Examining the weights the learned classifier assigned to each of the words in selected_words and gain intuition as to what the ML algorithm did for the data using these features

# In[60]:


coeff=sentiment_model2.coefficients


# In[62]:


coeff=coeff.sort('value',ascending=False)


# In[63]:


sentiment_model2.summary()


# ### Here we can find- 
# ##### The most positive weight has been given to the word-'love'  
# ##### The most negative weight has been given to the word- 'horrible'

# In[95]:


sentiment_model2.evaluate(test_set)


# In[75]:


top_prod2=sf[sf['name']=='Vulli Sophie the Giraffe Teether']


# In[90]:


top_prod2


# In[88]:


sentiment_model.evaluate(test_data)


# ## Majority Class Classifier

# In[116]:


train_set2,test_set2=top_prod2.random_split(0.8,seed=0)


# In[117]:


model3=tc.logistic_classifier.create(train_set2,features=selected_words,target='sentiment',validation_set=test_set2)


# In[118]:


model3.evaluate(test_set2)


# ### Comparison of 3 models based on their accuracy

# In[119]:


print(tc.evaluation.accuracy(test_data['sentiment'],sentiment_model.predict(test_data))*100,'%',tc.evaluation.accuracy(test_set['sentiment'],sentiment_model2.predict(test_set))*100,'%',tc.evaluation.accuracy(test_set2['sentiment'],model3.predict(test_set2))*100,'%')


# ##### Clearly, 1st model,i.e., the model used with all reviews and and words considered is better.

# In[103]:


top_prod2['predicted_sentiment']=sentiment_model.predict(top_prod2)
top_prod2


# In[122]:


model3.evaluate(test_data), sentiment_model2.evaluate(test_data), sentiment_model.evaluate(test_data)


# ## Understanding the difference between the classifiers created and their results.

# In[123]:


prod3=sf[sf['name']=='Baby Trend Diaper Champ']


# In[124]:


prod3


# ### Analyzing a particular random product

# In[126]:


prod3['predicted_sentiment']=sentiment_model.predict(prod3,output_type='probability')


# In[127]:


prod3=prod3.sort('predicted_sentiment',ascending=False)


# In[128]:


prod3


# In[130]:


prod4=sf[sf['name']=='Baby Trend Diaper Champ']
prod4['predicted_sentiment']=sentiment_model2.predict(prod4,output_type='probability')


# In[134]:


prod4=prod4.sort('predicted_sentiment',ascending=False)
prod4


# ### Finding the predicted sentiment probability of most positive reviewed random product according to 1st model

# In[133]:


sentiment_model2.predict(prod3[0:1], output_type='probability')


# In[136]:


prod3[0:1]['review']


# In[138]:


prod3[0:1]['word_count']


# In[137]:


prod3[0:1]


# In[ ]:




