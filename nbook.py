import numpy as np  
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn as sk

df = pd.read_csv('dataset_SCL.csv')

#1. Exploring Data 
# print(df.head(5))
# print(df.info())
# print(df['DIANOM'].describe())

#Check for missing values
# print(df.isna().sum())

#codes = used_cars['manufacturer_name'].cat.codes
#categories = used_cars['manufacturer_name']
#name_map = dict(zip(codes, categories))
#used_cars['manufaturer_code'].map(name_map)

#boolean coding
#Find all body_type that contains "van"
#used_cars['body_type'].str.contains("van", regex=False)
#used_cars["van_code"] = np.where(used_cars["body_type"].str.contains("van", regex=False), 1, 0)

#1 if Date-I is between Dec-15 and Mar-3, or Jul-15 and Jul-31, or Sep-11 and Sep-30, 0 otherwise.
def is_high_season(date):
    month = date.month
    day = date.day
    return (
        (month == 12 and day >= 15) or
        (month == 1 and day <= 3) or
        (month == 7 and day >= 15 and day <= 31) or
        (month == 9 and day >= 11 and day <= 30)
    )

def which_period_day(date):
    hour = date.hour
    if hour >= 5 and hour < 12:
        return 'morning'
    if hour > 12 and hour < 19:
        return 'afternoon'
    return 'night'

df['Fecha-I'] = pd.to_datetime(df['Fecha-I'])
df['Fecha-O'] = pd.to_datetime(df['Fecha-O'])

# df_synth = pd.DataFrame()
# df_synth['high_season'] = np.where(pd.to_datetime(df['Fecha-I']).map(is_high_season),1,0)
# df_synth['min_diff'] = (df['Fecha-I'].dt.hour - df['Fecha-O'].dt.hour)*60 + (df['Fecha-I'].dt.minute - df['Fecha-O'].dt.minute)
# df_synth['delay_15'] = np.where(df_synth['min_diff'].map(lambda x: x < 0 and np.abs(x) > 15), 1, 0)
# df_synth['period_day'] = pd.to_datetime(df['Fecha-I']).map(which_period_day)

# df_synth.to_csv('synthetic_features.csv', index=False)

df_synth = pd.read_csv('synthetic_features.csv')
df = pd.concat([df, df_synth], axis=1)

# print(df_synth.head(5))
# delay_mean_by_destination = df[df['min_diff'] < 0].groupby('TIPOVUELO')['min_diff'].value_counts()
# print(delay_mean_by_destination)

# plt.figure(figsize=(12, 6))
# sns.violinplot(x='DIANOM', y='min_diff', data=df[df['min_diff'] < 0])
# plt.title('Distribution of Flight Delays by Day of the Week')
# plt.xlabel('Day of the Week (DIA)')
# plt.ylabel('Delay in Minutes (Fecha-I - Fecha-O)')
# plt.show()

def plot_rate_delay(df, feature):
    delay_counts = df[df['delay_15'] == 1].groupby(feature)['delay_15'].count()
    on_time_counts = df[df['delay_15'] == 0].groupby(feature)['delay_15'].count()
    df_counts = pd.DataFrame({'delay': delay_counts, 'on_time': on_time_counts})

    plt.figure(figsize=(12, 6))
    sns.barplot(x=delay_counts.index, y=df_counts['delay'], color='red', label='Delayed Flights')
    sns.barplot(x=df_counts.index, y=df_counts['on_time'], color='black', bottom=df_counts['delay'])

    plt.ylabel('Count of Flights')
    plt.legend()
    plt.show()
    
    return

plot_rate_delay(df, 'DIANOM')