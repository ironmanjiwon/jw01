#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()
iris


# In[3]:


import pandas as pd

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['class'] = iris.target

df.tail()


# In[5]:


from sklearn.datasets import load_iris


# In[6]:


from sklearn.metrics import accuracy_score


# In[7]:


import numpy as np


# In[8]:


import pandas as pd


# In[9]:


iris = load_iris()


# In[10]:


x_train = iris.data[:-30]
y_train = iris.target[:-30]


# In[11]:


x_test = iris.data[-30:]
y_test = iris.target[-30:]


# In[12]:


print(y_train)


# In[13]:


print(y_test)


# In[14]:


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) 

x = np.arange(-10.0, 10.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.show()


# In[15]:


display(mglearn.plots.plot_logistic_regression_graph())


# In[ ]:




