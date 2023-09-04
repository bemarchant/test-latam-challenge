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
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 5))
    
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

def plot_roc_curve(y_probs_test, y_probs_train, y_test, y_train):
    fpr_test, tpr_test, thresholds = roc_curve(y_test, y_probs_test)
    fpr_train, tpr_train, thresholds = roc_curve(y_train, y_probs_train)
    roc_auc_test = auc(fpr_test, tpr_test)
    roc_auc_train = auc(fpr_train, tpr_train)

    plt.figure(figsize=(15, 5))
    plt.plot(fpr_train, tpr_train, color='gray', lw=2, label=f'Train ROC curve (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='black', lw=2, label=f'Test ROC curve (AUC = {roc_auc_test:.2f})')
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

def plot_feature_importances_top10(clf, feature_names):  
    plt.figure(figsize=(15, 5))  
    top=20
    feature_importances_dict = {}
    for i, name in enumerate(clf.feature_importances_):
        feature_importances_dict[i] = name

    sorted_feature_indices = sorted(feature_importances_dict,key=lambda k: feature_importances_dict[k], reverse=True)
    top_feature_indices = sorted_feature_indices[:top]
    top_feature_importances = [feature_importances_dict[i] for i in top_feature_indices]
    top_feature_names = [feature_names[i] for i in top_feature_indices]

    plt.barh(range(top), top_feature_importances, color='black')
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.yticks(np.arange(top), top_feature_names)
    plt.show()
    
    return