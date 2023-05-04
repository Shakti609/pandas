#!/usr/bin/env python
# coding: utf-8

# In[46]:


### Import libraries and packages
import numpy as np
import pandas as pd
# using data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[47]:


df=pd.read_csv("titanic-training-data.csv")


# In[48]:


df.head()


# In[49]:


df.head(10)


# In[50]:


df.tail()


# In[51]:


df.sample(10)


# In[52]:


df.dtypes


# In[53]:


df.info() # gets the total number of available obvservations


# In[54]:


df.shape


# In[55]:


df.isnull().sum() # gets the number of missing values 


# In[56]:


df.describe()


# In[57]:


df.describe(include='all')


# In[58]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[64]:


plt.hist(x=df["Age"],color="red")
plt.title("Distribution of age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[71]:


plt.hist(x=df["Sex"],color="yellow")
plt.title("Sex distribution",fontsize=20,color='green')
plt.xlabel("Sex",color='blue')
plt.ylabel("frequency",color='blue')
plt.show()


# In[82]:


sns.boxplot(x=df["Age"],color="pink")
plt.title("Distribution of age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[85]:


df["Sex"].value_counts().plot(kind="bar")
plt.title("Sex distribution")
plt.xlabel("Sex")
plt.ylabel("count")
plt.show()


# In[90]:


df["Sex"].value_counts().plot(kind="barh",color="orange")
plt.title("Sex distribution")
plt.xlabel("Sex")
plt.ylabel("count")
plt.show()


# In[89]:


df["Pclass"].value_counts().plot(kind="bar",color="green")
plt.title("Pclass distribution")
plt.xlabel("class")
plt.ylabel("count")
plt.show()


# In[104]:


from matplotlib import cm
emb=df["Embarked"].value_counts()
keys=emb.keys().to_list()
counts=emb.to_list()
cs=cm.Set1([2,4,6,8])
plt.pie(x=counts,labels=keys,autopct='%1.1f%%',colors=cs)
plt.show()


# In[105]:


plt.scatter(x="Age",y="SibSp",data=df)


# In[ ]:




