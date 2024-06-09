#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df_tracks=pd.read_csv(r'E:\spotify_datasets\tracks.csv')
df_tracks.head()


# In[4]:


#checking for the null values
pd.isnull(df_tracks).sum()


# In[5]:


df_tracks.info()


# In[6]:


#To find the least popular songs on spotify
sorted_df=df_tracks.sort_values('popularity',ascending = True).head(10)
sorted_df


# In[7]:


df_tracks.describe().transpose()


# In[8]:


most_popular=df_tracks.query('popularity>90',inplace=False).sort_values('popularity', ascending= False)
most_popular[:10]
#The sort_values is a part of the pandas library 


# In[9]:


df_tracks.set_index("release_date",inplace= True)
df_tracks.index=pd.to_datetime(df_tracks.index)
df_tracks.head()


# In[10]:


#To check rows of information from the dataset


# In[11]:


df_tracks[["artists"]].iloc[45]


# In[12]:


df_tracks["duration"]=df_tracks["duration_ms"].apply(lambda x : round(x/1000))
df_tracks.drop("duration_ms",inplace=True,axis=1)


# In[13]:


df_tracks


# In[14]:


#checking the values in duration column we get
df_tracks.duration.head(15)


# In[15]:


corr_df=df_tracks.drop(["key","mode","explicit"],axis=1).corr(method="pearson")
plt.figure(figsize=(14,6))
heatmap=sns.heatmap(corr_df,annot=True,fmt=".1g",vmin=-1,vmax=1,center=0,cmap="inferno",linewidths=1, linecolor="Black")
heatmap.set_title("Correlation HeatMap Between Variables .")
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation= 90)


# In[16]:


df_sample=df_tracks.sample(int(0.004*len(df_tracks)))
print(len(df_sample))


# In[17]:


plt.figure(figsize=(10,6))
sns.regplot(x="loudness",y="energy",data=df_sample,color="red").set(title="Energy vs Loudness Correlation")


# In[18]:


plt.figure(figsize=(10,6))
sns.regplot(x="popularity",y="danceability",data=df_sample,color="green").set(title="Danceability vs Popularity Correlation")


# In[19]:


df_tracks['dates']=df_tracks.index.get_level_values('release_date')
df_tracks.dates=pd.to_datetime(df_tracks.dates)
years=df_tracks.dates.dt.year


# In[20]:


pip install --user seaborn==0.11.0


# In[21]:


sns.displot(years,discrete=True,height=5,aspect=2,kind="hist").set(title="No of songs per year")


# In[22]:


total_dr=df_tracks.duration
fig_dims=(18,7)
fig, ax = plt.subplots(figsize=fig_dims)
fig=sns.barplot(x=years,y=total_dr,ax=ax,errwidth=False).set(title="Year vs Duration")
plt.xticks(rotation=90)


# In[26]:


total_dr=df_tracks.duration
sns.set_style(style="whitegrid")
fig_dims=(10,5)
fig, ax =plt.subplots(figsize=fig_dims)
fig=sns.lineplot(x=years,y=total_dr,ax=ax).set(title="Year vs Duration")
plt.xticks(rotation=60)


# In[27]:


df_genres=pd.read_csv("E:\spotify_datasets\SpotifyFeatures.csv")


# In[28]:


df_genres.head()


# In[30]:


plt.title("Duration of the songs in different Genres")
sns.color_palette("rocket",as_cmap=True)
sns.barplot(y='genre',x="duration_ms",data=df_genres)
plt.xlabel("Duration in ms")
plt.ylabel("Genres")


# In[31]:


#Spotify top 5 Genres by popularity
#The head function contains 10 as the argument as some Genres may be repetitive


# In[33]:


sns.set_style(style="darkgrid")
plt.figure(figsize=(10,5))
famous=df_genres.sort_values("popularity",ascending=False).head(10)
sns.barplot(y='genre',x='popularity',data=famous).set(title="Top 5 Genres by Popularity")


# In[ ]:




