#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # The dataset contains the tesla stock data from 2010-06-29 to 2020-02-03.

# TESLA has been on the rice recently, with a crazy +100% spike in the last30 days alone. with the history we can find out why?
# 
# 
# Stock Data includes Open,HIGH,Low,Adj close and Volume
# 
# 
# In stock trading, the high and low refer to the maximum and minimum prices in a given time period. Open are the prices at which a stock began and ended trading in the same period. volume is the total amount of trading activity . Adjusted values factor in corporate actions  such as dividends, stock splits, and new share issuance.
# 

# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/TSLA.csv')
df


# # varibales in Dataset:

# Date: Represents the date of the relevent Transaction Day.
# Open: Represents the initial share price of the relevant Trading Day.
# High: Represents the highest price of the relevant Trading Day.
# Low : it represents the lowest price of the relevant trading day.
# Close: It represents the closing price of the stock on the relevant trading day.
# Adj Close: Represents the adjusted closing price of the relevant trading day.
# Volume: it represents the trading volume information of the relevant trading day.

# In[3]:


df.head()


# Why the tesla stock increases rapidly?

# In[4]:


df.tail()


# it is giving the idea of last 10 rows.

# In[5]:


df.shape


# In[6]:


print("The dimension of the dataset:",df.shape)
print(f"\nThe column headers in the dataset: {df.columns}")


# This dataset contains 2416 rows and 7 columns. out of which 1 is target variable and remaning 6 are inpendent variables.

# In[7]:


print("Min. Date :",df.Date.min())
print("Max. Date:",df["Date"].max())


# In[8]:


df.dtypes


# There are three types of data (int64, float and object) present in the dataset.

# In[9]:


#Checking the null values
df.isnull().sum()


# As we can see there are no null values present in this dataset.

# In[10]:


df.info()


# This gives the brief about the dataset which includes indexing type, column type, no null values and memory usage.

# In[11]:


#Lests visualize it using heatmap
sns.heatmap(df.isnull())


# we can see clearly that there is no missing data present.

# In[12]:


"""""def fliprobo(df):
df["Date"] = pd.to_datetime(df["Date"])
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["date"] = df["Date"].dt.day
df["day"] = df["Date"].dt.dayofweek
df = df.drop(["Date"], axis=1, inplace=True)
return df

df_feat=fliprobo(df)
"""""


# In[13]:


#Converting the datatype of Date Column from object to datetime
df['Date'] = pd.to_datetime(df.Date, format = '%Y-%m-%d')
df.info()


# In[14]:


# setting Date as Dataframe Index
df.set_index('Date', inplace=True)


# The set_index() function is used to set the DataFrame index using existing columns.

# In[15]:


df


# In[16]:


# Checking number of unique values in each column
df.nunique().to_frame("No. of unique values")


# In[17]:


# Checking the value counts of each column
for i in df.columns:
    print(df[i].value_counts())
    print("\n")


# So, we will drop the close column since they contain same values.

# In[18]:


df.drop(["Adj Close"],axis=1, inplace=True)


# In[19]:


#i= df.drop(["Adj Close"],axis=1)


# In[20]:


df.head()


# Checking Duplicate values in DataFrame

# In[21]:


print("Total Duplicate Rows are",df.duplicated().sum())


# # Description of dataset

# In[22]:


# Statistical summary of numerical volumns
df.describe()


# This gives the statistical information of the numerical columns.The summary of the dataset looks perfect since there is no negative/invalid values present.
# 
# From the above description we can observation the following:
# 
# The counts of all the columns are same which means there are no missing values in the dataset.
# 
# The median(50%) value is greater than the mean in open, high, low, close columns which means the data is skewed to left in these columns.
# 
# By summarizing the data we can observe there is a huge difference between 75% and max hence there are outliers present int the data.
# 
# we can also notice the Standard deviation, minimum value, 25% percentile values from this describe method.
# 

# In[23]:


plt.figure(figsize=(22,10))
sns.heatmap(df.describe(),annot=True,fmt='0.2f',linewidth=0.2,linecolor='black',cmap='Spectral')
plt.xlabel('Figure',fontsize=14)
plt.title('Descriptive',fontsize=20)
plt.show()


# In[24]:


sns.lmplot(x='Open',y='Close',data=df,palette='colorblind')


# Open has a positive correlation with the close.

# In[25]:


sns.lmplot(x='Low',y='Close',data=df,palette='colorblind')


# Low has positive correlation with close.

# In[26]:


sns.lmplot(x='Volume',y='Close',data=df,palette='colorblind')


# Volume has a possitive correlation with close.

# In[27]:


sns.lmplot(x='Volume',y='High',data=df,palette='colorblind')


# Volume has possitive correlation with the high.

# In[28]:


plt.figure(figsize=(20,25))
p=1
for i in df:
    if p<=17:
        plt.subplot(5,4,p)
        sns.regplot(x='Close',y=i,data=df,color='r')
        plt.xlabel("Close")
        plt.ylabel(i)
    p+=1
    
    
plt.show()   


# In[29]:


plt.figure(figsize=(20,25))
p=1
for i in df:
    if p<=17:
        plt.subplot(5,4,p)
        sns.scatterplot(x='Close',y=i,data=df,color='r')
        plt.xlabel("Close")
        plt.ylabel(i)
    p+=1

plt.show()
         
   


# From this figure we can identify that all the features except volume have a high positive liner relarionship with the target variable.

# In[30]:


sns.pairplot(data=df, palette = "Dark2")


# The pairplot gives the pairwise relation between the features. on the diagonal we can notice the distribution plots.
# The features Low, High and Open have strong liner realtion with each other.

# In[31]:


plt.figure(figsize = (18,6))
plt.plot(df.Close, label = 'closing price')
plt.ylabel("Stock price")
plt.xlabel("Time")
plt.title("Tesla Stock Price")
plt.show()


# In[32]:


plt.figure(figsize=(15,8))
sns.lineplot(data=df.iloc[:,:-1])
plt.ylabel("Stock Value")
plt.title("The General Trend of all values")


# In[33]:


# # Lets check the outliers by plotting boxplot.

plt.figure(figsize=(20,25))
p=1
for i in df:
    if p<=13:
        plt.subplot(5,4,p)
        sns.boxplot(df[i], palette = "Set2_r")
        plt.xlabel(i)
        
    p+=1
    
plt.show()


# So we have found outlier in all columns.
# so, removing the outliers using zscore and IQR techniques before building the model and selected best one.

# In[34]:


from scipy.stats import zscore
out_features=df[['Open','High','Low','Volume']]
z=np.abs(zscore(out_features))
z


# In[35]:


#threeshold=3
np.where(z>3)


# In[36]:


z.iloc[723,3]


# In[37]:


# Now removing the data Zscore and creating new DF
df1 = df[(z<3).all(axis=1)]

df1.shape


# In[38]:


"""
x- has indpendent variables
y- target variable
z=np.abs(zscore(x))
x1= x[(z<3).all(axis=1)]

y1=y[(z<3).all(axis=1)]

df=df[(z<3).all(axis=1)]
"""


# In[39]:


#Shape of old and New Dataframe
print("Old Data Frame- ",df.shape[0])
print("New Data Frame- ",df1.shape[0])


# In[40]:


print("Data Loss Percentage-",((df.shape[0]-df1.shape[0])/df.shape[0])*100)


# In[41]:


df1


# it is affordable loss of data now we will check IQR method

# In[42]:


# 1st quantile
Q1=out_features.quantile(0.25)

# 3rd quantile
Q3=out_features.quantile(0.75)

# IQR
IQR=Q3 - Q1

df2=df[~((df < Q1 -1.5 * IQR)) | (df> (Q3 + 1.5 * IQR)).any(axis=1)]


# In[43]:


df2.shape


# In[44]:


print("Data Loss Percentage After removing outliers with IQR method- ",((df.shape[0]-df2.shape[0])/df.shape[0])*100)


# # Checking how the data has been distributed in each column

# In[47]:


plt.figure(figsize=(20,25), facecolor='green')
plotnumber = 1

for column in df:
    if plotnumber<=18:
        ax = plt.subplot(6,4,plotnumber)
        sns.distplot(df[column],color='b')
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.show()
    


# From the above distribution plots we can notice that the data almost looks nomrmal in all the columns in all the columns except Volume.

# # Checking for skewness

# In[48]:


df.skew()


# In[ ]:




