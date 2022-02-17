#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot
from bokeh.models.widgets import Tabs,Panel 
import pickle
output_notebook()


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/Ainagbolahan/HDSC-21-SCIPY-CAPSTONE-PROJECT/main/vgsales.xls')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


years=df.groupby('Year')['Global_Sales'].sum()
years.plot()


# In[7]:


# dropping all missing values

df = df.dropna()


# In[8]:


df.isnull().sum()


# In[9]:


# checking for duplicate rows

df.duplicated().sum()


# In[10]:


df['Year'] = df['Year'].astype(np.int64)


# In[11]:


df.head()


# In[12]:


df.Genre.unique()


# In[13]:


factors = list(df.Genre.unique()) #color mapping genre of games
colors = ["red","green","blue","black","orange","brown","grey","purple","yellow","white","pink","peru"]
mapper = CategoricalColorMapper(factors = factors,palette = colors)
plot =figure()
plot.circle(x= "Year",y = "Global_Sales",source=ColumnDataSource(df),color = {"field":"Genre","transform":mapper})
show(plot)


# In[14]:


df.groupby('Genre').Global_Sales.sum().plot(kind='pie',figsize=(8,8));


# In[15]:


ax=df.groupby('Year').Global_Sales.sum().plot(kind='bar', figsize=(12,5));
ax.set_xlabel('Year');
ax.set_ylabel('Global Sales');


# In[16]:


ax=df.groupby('Platform').Global_Sales.sum().plot(kind='bar', figsize=(12,5),color= 'orange');
ax.set_xlabel('Platform');
ax.set_ylabel('Global Sales');


# In[17]:


ax=df.groupby('Genre').Global_Sales.sum().plot(kind='bar', figsize=(12,5));
ax.set_xlabel('Genre');
ax.set_ylabel('Global Sales');


# In[18]:


df.loc[:, ['EU_Sales','NA_Sales','JP_Sales','Other_Sales', 'Genre']].groupby('Genre').sum().plot(kind='bar', figsize=(10,4))


# In[19]:


publishers = df[['Publisher','Year','Global_Sales','EU_Sales','NA_Sales','JP_Sales']]
publishers.head()


# In[20]:


publishers.describe()


# In[21]:


df.head()


# In[22]:


df2 = df.copy()


# In[23]:


import seaborn as sns
plt.figure(figsize = (15, 8))
sns.heatmap(df2.corr(), annot = True)


# In[24]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df2['Platform'] = encoder.fit_transform(df2.Platform)
df2['Genre'] = encoder.fit_transform(df2.Genre)
df2['Publisher'] = encoder.fit_transform(df2.Publisher)
df2['Name'] = encoder.fit_transform(df2.Name)

df2.head()


# In[25]:


df2.head()


# In[26]:


import seaborn as sns
plt.figure(figsize = (15, 5))
sns.heatmap(df2.corr(), annot = True)


# ### LINEAR REGRESSION (Rank and Name Columns dropped) ###

# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df2.drop(['Global_Sales', 'Rank', 'Name','NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis = 1)
y = df2['Global_Sales']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standard_X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(standard_X, y, test_size = 0.3, random_state=4)
linrm = LinearRegression()
linrm.fit(X_train, y_train)


# In[28]:


#Prediction and Evaluation
coeff_df = pd.DataFrame(linrm.coef_, X.columns, columns=['Cofficient'])
print(coeff_df)
pred = linrm.predict(X_test)


# In[29]:


# Comparing Actual vs Predicted
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': pred}) 
pred_df


# In[30]:


from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, pred))
print('MSE: ', metrics.mean_squared_error(y_test, pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print('rss: ', np.sum(np.square(y_test - pred)))
metrics.explained_variance_score(y_test,pred)


# ### LINEAR REGRESSION (With only Rank dropped) ###

# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df2.drop(['Global_Sales', 'Rank','NA_Sales','EU_Sales','JP_Sales','Other_Sales'], axis = 1)
y = df2['Global_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=4)

linrm = LinearRegression()
linrm.fit(X_train, y_train)


# In[32]:


predict = linrm.predict(X_test)


# In[33]:


from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, predict))
print('MSE: ', metrics.mean_squared_error(y_test, predict))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predict)))
print('rss: ', np.sum(np.square(y_test - predict)))
metrics.explained_variance_score(y_test,predict)


# ### LASSO REGRESSION ###

# In[34]:


from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.001)
lasso.fit(X_train, y_train)
pred_lasso = lasso.predict(X_test)

from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, pred_lasso))
print('MSE: ', metrics.mean_squared_error(y_test, pred_lasso))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred_lasso)))

metrics.explained_variance_score(y_test,pred)


# ### RIDGE REGRESSION ###

# In[35]:


# RIDGE REGRESSION

from sklearn.linear_model import Ridge

ridge = Ridge(alpha = 0.5)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)

from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, pred))
print('MSE: ', metrics.mean_squared_error(y_test, pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[36]:


metrics.explained_variance_score(y_test,pred)


# ### RANDOM FOREST REGRESSION ###

# In[37]:


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)

#prediction and evaluation
rfr_pred = rfr.predict(X_test)


from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, rfr_pred))
print('MSE: ', metrics.mean_squared_error(y_test, rfr_pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))
print('explained variance score: ', metrics.explained_variance_score(y_test,rfr_pred))


# In[38]:


rfr_df = pd.DataFrame({'Actual': y_test, 'Predicted': rfr_pred})
rfr_df


# In[41]:


X.head()


# In[42]:


pickle.dump(rfr, open('model.pkl', 'wb'))


# In[43]:


model = pickle.load(open('model.pkl', 'rb'))


# In[48]:


print(model.predict([[2002,2,2019,2, 200]]))


# In[ ]:




