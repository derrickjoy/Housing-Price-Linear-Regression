#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML

plt.style.use("ggplot")


# # Import the Data

# In[3]:


train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")


# In[4]:


train.columns


# In[5]:


train.head()


# In[6]:


train.info()


# # Many missing data in these columns.

# In[7]:


train.isna().sum().sort_values(ascending = False)[:6].index


# In[8]:


train = train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1)


# # Continuous Variables.

# In[9]:


continuos_variable = train.columns[train.nunique() > 20]
continuos_variable = train[continuos_variable]


# In[10]:


continuos_variable.describe().T


# # Categorical Data

# In[11]:


Categorical = train.columns[train.nunique() <= 15]
Categorical = train[Categorical]


# In[12]:


Categorical.describe().T


# # Exploratory Data Analysis

# In[13]:


corrmat = train.corr().nlargest(20, "SalePrice")
corrmat = corrmat.T.sort_values("SalePrice", ascending = False)[:20]
mask = np.triu(np.ones_like(corrmat))
plt.figure(figsize = (10, 10))
sns.heatmap(data = corrmat, mask = mask, annot = True, square = True, cmap = "Blues", vmax = .8, annot_kws = {"fontsize" : 8})

