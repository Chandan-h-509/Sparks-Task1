#!/usr/bin/env python
# coding: utf-8

# # TASK1 of GRIP@The Sparks Foundation
# # CHANDAN H

# # Objective:To predict the percentage of a student based on the number of hours studied using Linear Regression.

# In[2]:


#Import
import pandas as pd
import numpy as np
import seaborn as sns


# In[3]:


df=pd.read_csv("TASK1.txt")
df


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.corr()


# In[8]:


sns.heatmap(df.corr())


# In[9]:


#Plotting
sns.jointplot(x='Hours',y='Scores',data=df,kind="reg")


# In[10]:


#Variabl partitioning
X=df["Hours"]
Y=df["Scores"]


# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[13]:


#Model Training
import statsmodels.api as sm
sm.add_constant(X)


# In[14]:


model=sm.OLS(Y_train,X_train).fit()


# In[15]:


model.summary()


# In[22]:


#Comparison
Y_pred = model.predict(X_test)


# In[23]:


df1 = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
df1


# In[24]:


#Calculating Error
from sklearn import metrics  
print('Mean Absolute Error=', 
      metrics.mean_absolute_error(Y_test, Y_pred)) 


# In[28]:


#Objective
hours=9.25
score=model.predict(hours)
score[0]

