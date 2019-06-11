import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing



cd \Users\Mountassir Youssef\Desktop\challenge\data

#data acquisition

train=pd.read_csv('train.csv')
train.columns=['Unnamed: 0', 'ID_1', 'ID_2', 'ID_T', 'ID_R', 'RESULT','DATE']
train.head()
stat=pd.read_csv('stats.csv')
stat.head()
stat.columns
#first player data
stat_1=stat[['ID1','RPW_1', 'RPWOF_1']]
stat_1.columns=['ID_1', 'RPW_1', 'RPWOF_1']
stat_1.head()
#second player data
stat_2=stat[['ID2','RPW_2', 'RPWOF_2']]
stat_2.columns=['ID_2', 'RPW_2', 'RPWOF_2']
stat_2.head()


#merging table into one table
merge_1=pd.merge(left=train.head(2000),right=stat_1.head(2000),left_on='ID_1',right_on='ID_1',how='inner')
merge_1.head()

merge_2=pd.merge(left=merge_1.head(2000),right=stat_2.head(2000),left_on='ID_2',right_on='ID_2',how='inner')
merge_2.head()

merge = merge_2[['ID_1','ID_2','ID_T','ID_R','RPW_1','RPWOF_1','RPW_2','RPWOF_2']]
merge.head()
final_merge=merge_2[['ID_1','ID_2','ID_T','ID_R','RPW_1','RPWOF_1','RPW_2','RPWOF_2']]
final_merge.head()

final_merge.head()
final_merge.columns

#splitting inputs and output, standardization of data
X=final_merge[['ID_1', 'ID_2', 'ID_T', 'ID_R', 'RPW_1', 'RPWOF_1', 'RPW_2', 'RPWOF_2']].values
y=final_merge['RESULT'].values
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

#training
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X[0:1000], y[0:1000], test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
#KNN Method
from sklearn.neighbors import KNeighborsClassifier

#a loop to test which K is the best
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


#k=7 is the best

y_test_hat = neigh.predict(X_t)
y_test_hat
