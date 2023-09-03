import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

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

def plot_rate_delay(df, features):
    
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(10 * n_features, 5))
    
    if n_features == 1:
        axes = [axes]  # Convert to a list if there's only one feature
    for i, feature in enumerate(features):
        delay_counts = df[df['delay_15'] == 1].groupby(feature)['delay_15'].count()
        on_time_counts = df[df['delay_15'] == 0].groupby(feature)['delay_15'].count()
        df_counts = pd.DataFrame({'delay': delay_counts, 'on_time': on_time_counts})
        df_counts['delay'] = df_counts['delay'].fillna(0)
        df_counts['on_time'] = df_counts['on_time'].fillna(0)
        df_counts['delay_ratio'] = df_counts['delay'] / (df_counts['delay'] + df_counts['on_time'])

        sns.pointplot(x = df_counts.index, y = df_counts['delay_ratio'], color='black', ax=axes[i])
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)
        axes[i].set_ylabel('Delay Ratio')
        axes[i].set_title(f'Feature: {feature}')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

    return