# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:19:54 2021

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 19 15:58:58 2021

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 16 20:22:11 2021

@author: User
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score

#%%
df=pd.read_csv('covid.csv')
print(df.head())
#%%
print(df.isnull().sum())
#%%
covid= df.drop(['id','entry_date','date_symptoms','date_died'], axis = 1)
print(covid.head())

#%%
features = covid.columns
print(features)
#%%
feature= [x for x in features if x!= 'icu']
print(feature)
#%%
print(covid['covid_res'].value_counts().to_frame())
covid=covid[covid['covid_res']==1]
print(covid['covid_res'].value_counts().to_frame())
#%%
print(covid['icu'].value_counts().to_frame())
covid=covid[covid['icu']!=97]
covid=covid[covid['icu']!=99]
print(covid['icu'].value_counts().to_frame())

#%%

for column_name in feature:
    covid[column_name]= covid[column_name].apply(lambda x:0 if x!=1 else x)

print(covid.head())

#%%
x=covid[feature]
y=covid['icu']
#%%
icu = ['icu needed','icu not needed']
#%%
dt= DecisionTreeClassifier(min_samples_split = 100, criterion='entropy',max_depth=9)
#%%
rf=RandomForestClassifier(n_estimators=100,max_depth=9)   

#%%
scdf=cross_val_score(dt, x, y,cv=10)
scrf=cross_val_score(rf, x, y,cv=10)
print(sum(scdf)/len(scdf))
print(sum(scrf)/len(scrf))

#%%
train, test = train_test_split(covid, test_size = 0.25)
#%%
x_train = train[feature]
y_train = train['icu']

x_test = test[feature]
y_test = test['icu']

#%%
dt = dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

#%%

score = accuracy_score(y_test, y_pred) * 100
print("Accuracy using desicion Tree: ", round(score, 2), "%" )

#%%
    
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)  

#%% 
score1=accuracy_score(y_test, y_pred_rf)*100
print("Accuracy using random forest:",round(score1, 2), "%")   

#%%
plot_confusion_matrix(dt, x_test, y_test,cmap='PuBuGn',values_format='.1f')  
plt.title("Confusion matrix by decision tree")
plt.show() 

#%%
plot_confusion_matrix(rf, x_test, y_test,cmap='PuBuGn',values_format='.1f')  
plt.title("Confusion matrix by random forest")
plt.show()
#%%

tree.export_graphviz(dt, out_file="treeCicu.dot", feature_names=feature,  
                     class_names=icu,
                filled=True, rounded=True,
                special_characters=True)

#%%

graph = pydotplus.graphviz.graph_from_dot_file("treeCicu.dot")
graph.write_png('treeCicu.png')
             