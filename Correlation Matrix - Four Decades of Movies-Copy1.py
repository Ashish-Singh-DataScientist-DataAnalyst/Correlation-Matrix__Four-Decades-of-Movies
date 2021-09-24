#!/usr/bin/env python
# coding: utf-8

# # Correlation Matrix : Four Decades of Movies

# In[126]:


# First let's import the libraries and packages we will use in this project
# We can do this all now or as you need them


# In[127]:


import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)


# In[128]:


# Reading The File


# In[129]:


DataFrame = pd.read_csv('movies.csv')


# In[130]:


DataFrame.head()


# In[131]:


#looking for missing values i.e; null values


# In[132]:


DataFrame.isnull()


# In[133]:


DataFrame.isnull().sum()


# In[134]:


DataFrame.dtypes


# In[135]:


df1=DataFrame.replace(['<NA>'],np.NaN)


# In[136]:


df1


# In[137]:


#df1.fillna(0)


# In[138]:


#df1['budget'].fillna(1)


# In[139]:


#df1['gross'].fillna(1)


# In[140]:


df1


# In[141]:


DF=df1.dropna()


# In[142]:


DF


# In[143]:


DF.isnull().sum()


# In[144]:


for col in DataFrame.columns:
    Prct_Missing_Values = np.mean (DataFrame[col].isnull())
    print('{} _ {}%'.format(col, Prct_Missing_Values))


# In[145]:


DF.dtypes


# In[146]:


# Are there any Outliers?

DF.boxplot(column=['gross'])


# In[147]:


DF['budget'] = DF['budget'].astype(pd.np.int64)


# In[148]:


DF.dtypes


# In[149]:


DF['gross'] = DF['gross'].astype(pd.np.int64)


# In[150]:


DF.dtypes


# In[151]:


# Using scatter plot to Interpret our data

plt.scatter(x=DF['budget'], y=DF['gross'])

plt.show()


# In[152]:


sns.regplot(x="gross", y="budget", data=DF)


# In[ ]:





# In[153]:


DF.sort_values(by=['gross'], inplace=False, ascending=False)


# In[ ]:





# In[154]:


sns.regplot(x="score", y="gross", data=DF)


# In[155]:


DF.corr(method ='pearson')


# In[156]:


DF.corr(method ='kendall')


# In[157]:


DF.corr(method ='spearman')


# In[158]:


correlation_matrix = DF.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[ ]:





# In[159]:


DF['company'].drop_duplicates().sort_values(ascending=False)


# In[160]:


# Using factorize - this assigns a random numeric value for each unique categorical value

DF.apply(lambda x: x.factorize()[0]).corr(method='pearson')


# In[161]:


correlation_matrix = DF.apply(lambda x: x.factorize()[0]).corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[162]:


correlation_mat = DF.apply(lambda x: x.factorize()[0]).corr()

corr_pairs = correlation_mat.unstack()

print(corr_pairs)


# In[163]:


sorted_pairs = corr_pairs.sort_values(kind="quicksort")

print(sorted_pairs)


# In[164]:


# We can now take a look at the ones that have a high correlation (> 0.5)

strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]

print(strong_pairs)# Looking at the top 15 companies by gross revenue

CompanyGrossSum = DF.groupby('company')[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values('gross', ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[165]:


#DF['Year'] = df1['released'].astype(str).str[:4]


# In[166]:


DF['Year'] = df1['year'].astype(str).str[:4]
DF


# In[167]:


DF.groupby(['company', 'year'])[["gross"]].sum()


# In[168]:


CompanyGrossSum = DF.groupby(['company', 'year'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company','year'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[169]:


CompanyGrossSum = DF.groupby(['company'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[170]:


plt.scatter(x=DF['budget'], y=DF['gross'], alpha=0.5)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()


# In[171]:


DF


# In[172]:


DF_numerized = DF


for col_name in DF_numerized.columns:
    if(DF_numerized[col_name].dtype == 'object'):
        DF_numerized[col_name]= DF_numerized[col_name].astype('category')
        DF_numerized[col_name] = DF_numerized[col_name].cat.codes
        
DF_numerized


# In[173]:


DF_numerized.corr(method='pearson')


# In[174]:


correlation_matrix = DF_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[ ]:





# In[175]:


for col_name in DF.columns:
    if(DF[col_name].dtype == 'object'):
        DF[col_name]= DF[col_name].astype('category')
        DF[col_name] = DF[col_name].cat.codes


# In[176]:


DF


# In[177]:


#DF[cat_columns] = DF[cat_columns].apply(lambda x: x.cat.codes)


# In[178]:


sns.swarmplot(x="rating", y="gross", data=DF)


# In[179]:


sns.stripplot(x="rating", y="gross", data=DF)


# In[ ]:





# In[ ]:





# In[ ]:





# # Acknowledgements & Reproducibility

# In[1]:


#df1['gross'] = df1['gross'].astype('Int64')
#print (df1['gross'])


# In[2]:


#df1['budget'] = df1['budget'].astype('Int64')
#print (df1['budget'])


# In[13]:


#If any error occurs while converting dtype, the method inscribed and this alternative beloiw will resolve any syntax error.

#df['column name'] = df['column name'].astype(pd.np.int64)
#print (df['column name'])


# In[14]:


#DataFrame['Correct Release Year'] = DataFrame['released'].astype(str).str[8:13]


# In[15]:


# requires wrangling in release year data


# In[3]:


#df1.dropna(how="budget",thresh=7)


# In[4]:


#DataFrame.sort_values(by=['gross'], inplace=False, ascending=False)


# In[5]:


#DF = DataFrame.sort_values(by=['gross'], inplace=False, ascending=False)


# In[19]:


pd.set_option('display.max_rows', None)
#To get complete visibility


# In[6]:


#Filtering Any Possible Duplicates

#DataFrame['company'].sort_values(ascending=False)


# In[7]:


#DF

