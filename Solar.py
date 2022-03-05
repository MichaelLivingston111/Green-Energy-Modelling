# Import:
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import keras
from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# Import the required dataset:
data = pd.read_csv("Solar_Energy_Forecasting/Pasion et al dataset.csv")


# Our target variable is solar output, referred to here as 'PolyPwr'.
# Pop off solar output:
Solar_power = data.pop("PolyPwr")
df = data  # Rename


# Now, we need to transform the categorical variables using one-hot encoding (specifically location and season):
df_location_en = pd.get_dummies(df, columns=['Location'], drop_first=True)  # Location
df_loc_season_en = pd.get_dummies(df_location_en, columns=['Season'], drop_first=True)  # Season


# Now, we need to create cyclic features in the data from the month and hour data.

# Define time bounds in data (hours outside this range are not expected to generate much power): "Power cycle"
min_hour_of_interest = 10  # Min hour of interest
max_hour_of_interest = 15  # Max hour of interest


# Calculate time since beginning of power cycle:
df_loc_season_en['delta_hr'] = df_loc_season_en.Hour - min_hour_of_interest

# Create cyclic hour features
df_loc_season_en['sine_hr'] = np.sin((df_loc_season_en.delta_hr*np.pi/(max_hour_of_interest - min_hour_of_interest)))
df_loc_season_en['cos_hr'] = np.cos((df_loc_season_en.delta_hr*np.pi/(max_hour_of_interest - min_hour_of_interest)))

# Create cyclic month features
df_loc_season_en['sine_mon'] = np.sin((df_loc_season_en.Month - 1)*np.pi/11)
df_loc_season_en['cos_mon'] = np.cos((df_loc_season_en.Month - 1)*np.pi/11)

# Now, we can drop month and hour from our dataframe, as well as other redundant variables:
df_updated = df_loc_season_en.drop(['Hour', 'Month', 'delta_hr', 'YRMODAHRMI'], axis=1)


# Feature selection is the next important step. It will allow us to indetify the most influential variables in the
# dataset, and eliminate any variables that are of limited importance and may reduce model accuracy.

# Feature selection will be done using two methods (and comparing the results):
# 1: Univariate selection (SelectKBest technique)
# 2: Feature importance (Random Forest technique)

# Univariate Selection:
X = df_updated
Y = Solar_power

# feature extraction
selector = SelectKBest(score_func=f_classif, k='all').fit(X, Y)
scores = selector.scores_  # We now have a series of scores for each feature

# Feature names:
feature_names = list(X)

# Order the variables by feature importance:
feature_imp = pd.Series(scores, index=feature_names).sort_values(ascending=False)

# Visualize feature importance:
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()