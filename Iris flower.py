#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
iris = pd.read_csv("C:\\Users\\user\\Downloads\\iris\\IRIS.csv")
iris.head()


# In[22]:


iris.plot(kind="scatter", x="sepal_length", y="sepal_width")


# In[23]:


sns.jointplot(x="sepal_length", y="sepal_width", data=iris, size=5)


# In[26]:


sns.FacetGrid(iris, hue="species", size=5)    .map(plt.scatter, "sepal_length", "sepal_width")    .add_legend()


# In[27]:


sns.boxplot(x="species", y="petal_length", data=iris)


# In[28]:


sns.FacetGrid(iris, hue="species", size=6)    .map(sns.kdeplot, "petal_length")    .add_legend()


# In[31]:


sns.pairplot(iris, hue="species", size=3)


# In[32]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# In[33]:


X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[35]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:




