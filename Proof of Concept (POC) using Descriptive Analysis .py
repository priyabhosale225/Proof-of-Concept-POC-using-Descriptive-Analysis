#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


#Load data from csv file

def load_data(file_path):
    return pd.read_csv(file_path)


# In[3]:


def drop_column(df,column_name):
    if column_name in df.columns:
        df_dropped=df.drop(columns=[column_name])
        print(f"Column '{column_name}' has been dropped.")
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        df_dropped=df
    return df_dropped
            


# In[4]:


#Data preprocessing

def preprocess_data(df):
    df = drop_column(df,"Unnamed: 0")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)
        
    
    if 'Order Date' in df.columns:
        df['Order Date']=pd.to_datetime(df['Order Date'])
        
    
    
    return df


# In[5]:


#Identify dimensions (categorical) and measures (numerical) from the Dataset.

def identify_dimensions_and_measures(df):
    dimensions = []
    measures = []
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            measures.append(column)
        else:
            dimensions.append(column)
            
    return dimensions, measures
    


# In[6]:


def plot_descriptive_analysis(df, dimensions, measures):
    dimensions, measures = identify_dimensions_and_measures(df)
    
    print(f"Identified dimensions: {dimensions}")
    print(f"Identified measures: {measures}")


# In[7]:


#Generate histograms for numerical measures.

def plot_histograms(df, measures):
    for measure in measures:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[measure], kde=True, bins=30)
        plt.title(f'Histogram of {measure}')
        plt.xlabel(measure)
        plt.ylabel('Frequency')
        plt.show()


# In[8]:


#Generate boxplots for numerical measures.

def plot_boxplots(df, measures):
    for measure in measures:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=df[measure])
        plt.title(f'Boxplot of {measure}')
        plt.ylabel(measure)
        plt.show()


# In[9]:


#Generate scatter plots for pairs of numerical measures.

def plot_scatter_plots(df, measures):
    if len(measures) > 1:
        for i in range(len(measures)):
            for j in range(i + 1, len(measures)):
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=df[measures[i]], y=df[measures[j]])
                plt.title(f'Scatter plot of {measures[i]} vs {measures[j]}')
                plt.xlabel(measures[i])
                plt.ylabel(measures[j])
                plt.show()


# In[10]:


#Generate count plots for categorical dimensions.

def plot_count_plots(df, dimensions):
    for dimension in dimensions:
        plt.figure(figsize=(12, 8))
        sns.countplot(x=df[dimension], order=df[dimension].value_counts().index)
        plt.title(f'Count plot of {dimension}')
        plt.xlabel(dimension)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


# In[11]:


#Generate pair plots for numerical measures.

def plot_pair_plots(df, measures):
    if len(measures) > 1:
        pair_df = df[measures]
        plt.figure(figsize=(12, 12))
        sns.pairplot(pair_df)
        plt.show()


# In[12]:


#Generate a heatmap of the correlation matrix for numerical measures.

def plot_correlation_heatmap(df, measures):
    if len(measures) > 1:
        corr_matrix = df[measures].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()


# In[13]:


#Generate a time series plot of sales over time.

def plot_time_series(df):
    if 'Order Date' in df.columns and 'Sales' in df.columns:
        df.set_index('Order Date', inplace=True)
        plt.figure(figsize=(10, 6))
        df['Sales'].resample('M').sum().plot()
        plt.title('Monthly Sales Time Series')
        plt.xlabel('Order Date')
        plt.ylabel('Total Sales')
        plt.show()
        df.reset_index(inplace=True)


# In[14]:


#Generate a plot of sales by hour.

def plot_sales_by_hour(df):
    if 'Hour' in df.columns and 'Sales' in df.columns:
        plt.figure(figsize=(10, 6))
        df.groupby('Hour')['Sales'].sum().plot(kind='bar')
        plt.title('Total Sales by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Total Sales')
        plt.show()


# In[15]:


def main(file_path):
    df = load_data(file_path)
    df = preprocess_data(df)
    dimensions, measures = identify_dimensions_and_measures(df)
    
    
    print(f"Identified dimensions (categorical): {list(dimensions)}")
    print(f"Identified measures (numerical): {list(measures)}")
    
    
    plot_histograms(df, measures)
    plot_boxplots(df, measures)
    plot_scatter_plots(df, measures)
    plot_count_plots(df, dimensions)
    plot_pair_plots(df, measures)
    plot_correlation_heatmap(df, measures)
    plot_time_series(df)
    plot_sales_by_hour(df)


# In[ ]:


file_path = "D:/DATA SCIENCE/Python/Notes/Datasets/Sales Data.csv"  # Replace with your file path
main(file_path)


# In[ ]:




