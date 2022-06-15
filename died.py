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
covid= df.drop(['id','entry_date','date_symptoms'], axis = 1)
print(covid.head())

#%%
features = covid.columns
print(features)
#%%
feature= [x for x in features if x!= 'date_died']
print(feature)

#%%
covid['date_died']= df['date_died'].apply(lambda x:0 if x=='9999-99-99' else 1)
for column_name in feature:
    covid[column_name]= covid[column_name].apply(lambda x:0 if x!=1 else x)

print(covid.head())

#%%
x=covid[feature]
y=covid['date_died']
#%%
died = ['not dead','dead']
#%%
dt= DecisionTreeClassifier(min_samples_split = 100, criterion='entropy',max_depth=9)
#%%
rf=RandomForestClassifier(n_estimators=100,max_depth=9)   
 
#%%
scdf=cross_val_score(dt, x, y,cv=5)
scrf=cross_val_score(rf, x, y,cv=5)
print(sum(scdf)/len(scdf))
print(sum(scrf)/len(scrf))

#%%
train, test = train_test_split(covid, test_size = 0.25)
#%%
x_train = train[feature]
y_train = train['date_died']

x_test = test[feature]
y_test = test['date_died']

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

tree.export_graphviz(dt, out_file="tree.dot", feature_names=feature,  
                     class_names=died,
                filled=True, rounded=True,
                special_characters=True)

#%%

graph = pydotplus.graphviz.graph_from_dot_file("tree.dot")
graph.write_png('tree.png')
             