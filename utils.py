import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc

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
    fig, axes = plt.subplots(n_features, 1, figsize=(15 * n_features, 5))
    
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

def plot_roc_curve(y_probs, y_test):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    
    return

def plot_feature_importances(clf, feature_names):
    
    c_features = len(feature_names)
    plt.barh(range(c_features), clf.feature_importances_)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.yticks(np.arange(c_features), feature_names)
    
    return