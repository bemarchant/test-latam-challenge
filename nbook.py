import numpy as np  
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn as sk

from utils import *
df = pd.read_csv('dataset_SCL.csv')
# print(sorted(df['OPERA'].unique()))

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

# plot_rate_delay(df, ['Des-I','DIANOM','MES','Emp-I'])
# plot_rate_delay(df, ['Des-I'])
# plot_rate_delay(df, 'high_season')
# plot_rate_delay(df, 'TIPOVUELO')

delay_counts = df[df['delay_15'] == 1].groupby('high_season')['delay_15'].count()
on_time_counts = df[df['delay_15'] == 0].groupby('high_season')['delay_15'].count()
df_counts = pd.DataFrame({'delay': delay_counts, 'on_time': on_time_counts})
df_counts['delay'] = df_counts['delay'].fillna(0)
df_counts['on_time'] = df_counts['on_time'].fillna(0)
df_counts['delay_ratio'] = df_counts['delay'] / (df_counts['delay'] + df_counts['on_time'])

# machine learning 1
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

predictors = ['Des-I','Emp-I','DIANOM','TIPOVUELO','high_season','period_day']
target = ['delay_15']

X = df[predictors]
y = df[target]
X_encoded = pd.get_dummies(X, columns=['Des-I', 'Emp-I', 'DIANOM', 'TIPOVUELO', 'period_day'])
features_encoded = X_encoded.columns

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# X_train = X_encoded.iloc[:-15000]
# y_train = y.iloc[:-15000]
# X_test = X_encoded.iloc[-15000:]
# y_test = y.iloc[-15000:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#LogisticRegression
# from sklearn.linear_model import LogisticRegression
# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(X_train, y_train)

# threshold = 0.5

# y_pred_prob = log_reg.predict_proba(X_train)
# # y_pred = (y_pred_prob[:, 1] >= threshold).astype(int)
# y_pred = log_reg.predict(X_train)
# accuracy = accuracy_score(y_train, y_pred)
# print(f"Accuracy: {accuracy}")
# print("\nClassification Report:")
# print(classification_report(y_train, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_train, y_pred))

# plot_roc_curve(y_pred_prob[:,1], y_train)

#RandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
# y_pred = rf_model.predict(X_train)
# accuracy = accuracy_score(y_train, y_pred)
# print(f"Accuracy: {accuracy}")
# print("\nClassification Report:")
# print(classification_report(y_train, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_train, y_pred))


## DecisionTreeClassifier
# from sklearn.tree import DecisionTreeClassifier
# dt_clf = DecisionTreeClassifier(random_state=42)
# dt_clf.fit(X_train, y_train)
# y_pred = dt_clf.predict(X_train)
# y_pred_prob = dt_clf.predict_proba(X_train)

# accuracy = accuracy_score(y_train, y_pred)
# print(f"Accuracy: {accuracy}")
# print("\nClassification Report:")
# print(classification_report(y_train, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_train, y_pred))
# plot_feature_importances_top10(dt_clf, features_encoded)

# y_prob_train = dt_clf.predict_proba(X_train)
# y_prob_test = dt_clf.predict_proba(X_test)
# plot_roc_curve(y_prob_test[:,1], y_prob_train[:,1], y_test, y_train)

#GradientBoostingClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# gb_clf = GradientBoostingClassifier(random_state=42)
# gb_clf.fit(X_train, y_train)
# y_pred = gb_clf.predict(X_train)
# accuracy = accuracy_score(y_train, y_pred)
# print(f"Accuracy: {accuracy}")
# print("\nClassification Report:")
# print(classification_report(y_train, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_train, y_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [20,30],
    'max_depth': [2,10,20],
    'min_samples_split': [2,4,8],
    'min_samples_leaf': [2,4,8],
    'class_weight': ['balanced'],
}

rf_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train.values.ravel())

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_prob = best_model.predict_proba(X_test)
y_pred = (y_prob[:,1] >= 0.50).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n Best parameters")
print(best_params)

y_prob_train = best_model.predict_proba(X_train)
y_prob_test = best_model.predict_proba(X_test)
plot_roc_curve(y_prob_test[:,1], y_prob_train[:,1], y_test, y_train)
