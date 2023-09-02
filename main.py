import numpy as np  
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn as sk

df = pd.read_csv('dataset_SCL.csv')

#1. Exploring Data 
print(df.head(5))
print(df.info())
print(df['DIANOM'].describe())

#Check for missing values
print(df.isna().sum())

#codes = used_cars['manufacturer_name'].cat.codes
#categories = used_cars['manufacturer_name']
#name_map = dict(zip(codes, categories))
#used_cars['manufaturer_code'].map(name_map)

#boolean coding
#Find all body_type that contains "van"
#used_cars['body_type'].str.contains("van", regex=False)
#used_cars["van_code"] = np.where(used_cars["body_type"].str.contains("van", regex=False), 1, 0)

