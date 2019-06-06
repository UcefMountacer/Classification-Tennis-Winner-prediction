import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import random


# In[2]:


cd \Users\Mountassir Youssef\Desktop\challenge\data


# In[3]:


train=pd.read_csv('train.csv')


# In[4]:


train.columns=['Unnamed: 0', 'ID_1', 'ID_2', 'ID_T', 'ID_R', 'RESULT',
       'DATE']


# In[5]:


train.head()


# In[6]:


stat=pd.read_csv('stats.csv')
stat.head()


# In[7]:


stat.columns


# In[8]:


stat_1=stat[['ID1','RPW_1', 'RPWOF_1']]


# In[9]:


stat_1.columns=['ID_1', 'RPW_1', 'RPWOF_1']
stat_1.head()


# In[10]:


stat_2=stat[['ID2','RPW_2', 'RPWOF_2']]


# In[11]:


stat_2.columns=['ID_2', 'RPW_2', 'RPWOF_2']
stat_2.head()


# In[12]:


test=pd.read_csv('test.csv')
test.columns=['ID_1', 'ID_2', 'ID_T', 'ID_R', 'DATE']


# In[13]:


test.head()


# In[14]:


tour=pd.read_csv('tour.csv')
tour_mod=tour[['ID_T','DATE_T','RANK_T']]


# In[15]:


tour_mod.columns=['ID_T', 'DATE', 'RANK_T']
tour_mod.head()


# In[16]:


player=pd.read_csv('players.csv')


# In[17]:


player.head()


# In[18]:


player_mod=player[['ID_P','NAME_P']]


# In[19]:


player_mod.head()


# In[20]:


merge_1=pd.merge(left=train.head(2000),right=stat_1.head(2000),left_on='ID_1',right_on='ID_1',how='inner')
merge_1.head()


# In[21]:


merge_t_1=pd.merge(left=test.head(2000),right=stat_1.head(2000),left_on='ID_1',right_on='ID_1',how='inner')
merge_t_1.head()


# In[22]:


merge_t_2=pd.merge(left=merge_t_1.head(2000),right=stat_2.head(2000),left_on='ID_2',right_on='ID_2',how='inner')
merge_t_2.head()


# In[34]:


merge = merge_t_2[['ID_1','ID_2','ID_T','ID_R','RPW_1','RPWOF_1','RPW_2','RPWOF_2']]


# In[35]:


merge.head()


# In[28]:


merge_t=merge_t_2[['ID_1','ID_2','ID_T','ID_R','RPW_1','RPWOF_1','RPW_2','RPWOF_2']]
merge_t.head()


# In[36]:


#for i in range (len(merge)):
#    merge.iat[i,8]=str(random.randint(1,2))


# In[37]:


merge.head()


# In[38]:


merge.columns


# In[40]:


X=merge[['ID_1', 'ID_2', 'ID_T', 'ID_R', 'RPW_1', 'RPWOF_1', 'RPW_2', 'RPWOF_2']].values


# In[39]:


X_t=merge_t[['ID_1', 'ID_2', 'ID_T', 'ID_R', 'RPW_1', 'RPWOF_1', 'RPW_2', 'RPWOF_2']].values


# In[ ]:


#y=merge['RESULT'].values
#y


# In[ ]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# X_t is the array from test data frame on which we want to test our model
X_t = preprocessing.StandardScaler().fit(X_t).transform(X_t.astype(float))


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


#X_train, X_test, y_train, y_test = train_test_split( X[0:1000], y[0:1000], test_size=0.2, random_state=4)


# In[ ]:


print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:





# In[ ]:



k = 1
while k<11:
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = 7).fit(X_train,y_train)
    neigh
    yhat = neigh.predict(X_test)
    yhat
    from sklearn import metrics
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
    k=k+1


# In[ ]:


#over fitting problem
#train must be modified 
#test with random numbers 
#k=7 is the best


# In[ ]:


y_test_hat = neigh.predict(X_t)
y_test_hat


# In[ ]:


test['R']= pd.Series(y_test_hat[0:6097], index=test.index)
test


# In[ ]:


test.to_csv(r'\Users\Mountassir Youssef\Desktop\challenge\data\ results.csv')

