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
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density  # adds projection='scatter_density'
from scipy.stats import kde

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

# Now, we need to remove all the location specific data (i.e. specific region names). This is different from the
# previous file -  we want to build a model that can be extrapolated into new regions.
df_updated = df.drop(['YRMODAHRMI', 'Location'], axis=1)


# Define time bounds in data (hours outside this range are not expected to generate much power): "Power cycle"
min_hour_of_interest = 10  # Min hour of interest
max_hour_of_interest = 15  # Max hour of interest


# One hot encode 'Season':
df_updated = pd.get_dummies(df_updated, columns=['Season'], drop_first=True)  # Season


# Calculate time since beginning of power cycle:
df_updated['delta_hr'] = df_updated.Hour - min_hour_of_interest

# Create cyclic hour features
df_updated['sine_hr'] = np.sin((df_updated.delta_hr*np.pi/(max_hour_of_interest - min_hour_of_interest)))
df_updated['cos_hr'] = np.cos((df_updated.delta_hr*np.pi/(max_hour_of_interest - min_hour_of_interest)))

# Create cyclic month features
df_updated['sine_mon'] = np.sin((df_updated.Month - 1)*np.pi/11)
df_updated['cos_mon'] = np.cos((df_updated.Month - 1)*np.pi/11)


# Now, we can drop month and hour from our dataframe, as well as other redundant variables:
df_variables = df_updated.drop(['Hour', 'Month', 'delta_hr', 'Longitude', 'Pressure'], axis=1)

# We only have latitude, time measurements, and a series of environmental variables now. These are the necessary
# inputs for this project.


# Feature selection is the next important step. It will allow us to indetify the most influential variables in the
# dataset, and eliminate any variables that are of limited importance and may reduce model accuracy.
# Feature selection will be done using two Univariate selection:

# Univariate Selection:
X = df_variables
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


# Need to split the data into training and testing sets to build and test the model:
x_train1, x_test1, y_train1, y_test1 = train_test_split(df_variables, Solar_power, test_size=0.2, random_state=0)
print("Train data has {} data points, test data has {} data points" .format(x_train1.shape[0], x_test1.shape[0]))


# CREATE THE NEURAL NETWORK:

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(df_variables))

# Model architecture:
model = keras.Sequential([
    normalizer,
    layers.Dense(320, input_dim=x_train1.shape[1], activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(280, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(240, activation='tanh'),
    layers.Dense(1),
])

# Compile the model:
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss='mae',
    metrics=['mean_absolute_error']
)
model.build()
model.summary()


# Train the model(s):
num_epochs = 15
batch_size = 4000

history_1 = model.fit(x_train1, y_train1, epochs=num_epochs, validation_split=0.2)


# Evaluate the model:
loss1, mae1 = model.evaluate(x_test1, y_test1, verbose=2)
print(mae1)


# ASSESS THE PERFORMANCE OF THE ALGORITHM: Visualizing its accuracy and loss rate over each epoch will give us
# insight into whether or not the model is over/under fitting the data:

# Summarize history for loss: Solar predictions
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Make the predictions:
Solar_predictions = model.predict(x_test1)


# Plot the predictions:
plt.scatter(y_test1, Solar_predictions, alpha=0.1, cmap='jet')
lims = [0, 35]
plt.xlim(lims)
plt.ylim(lims)


# R2 score
r2_score(y_test1.ravel(), Solar_predictions)


# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
SP = Solar_predictions.reshape(-1)  # reshape for plotting
nbins = 300
k = kde.gaussian_kde([y_test1, SP])
xi, yi = np.mgrid[y_test1.min():y_test1.max():nbins * 1j, SP.min():SP.max():nbins * 1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Make the plot
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='jet')
plt.show()

