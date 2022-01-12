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


# DESCRIPTION: This dataset assesses the heating load and cooling load requirements of buildings (energy efficiency)
# as a function of a variety of building parameters. This is a regression porblem that can be optimized by using a
# sequential neural network.


# The parameters (predictors) are labelled X1-X8, as follows:
# X1 Relative Compactness
# X2 Surface Area
# X3 Wall Area
# X4 Roof Area
# X5 Overall Height
# X6 Orientation
# X7 Glazing Area
# X8 Glazing Area Distribution

# There are two target variables we are aiming to predict:
# y1 Heating Load
# y2 Cooling Load


# UPLOAD DATA: (Available from http://archive.ics.uci.edu/ml/datasets/Energy+efficiency#)
raw_data = pd.read_csv("ENB2012_data.csv")


# ISOLATE THE TARGET VARIABLES:
Target_1 = raw_data.pop("Y1")  # Heating load
Target_2 = raw_data.pop("Y2")  # Cooling load
df = raw_data  # Rename


# DATA CLEANING:
# Remove/replace Nans:
df.isnull().sum()  # No Nan values.

# Convert all data types to integers or floating points:
df.dtypes  # All variables are correct types.


# FEATURE SELECTION: We need to filter out any junk variables that may be unrelated ot our target variables in order
# to improve the accuracy of the model downstream. However, given the fact that we only have 8 variables I will skip
# this step for now. If necessary, I will use feature selection (select K best method using ANOVA F statistic) to
# hopefully improve the forecasting accuracy.


# TRAINING AND TESTING SETS:

# Target 1:
x_train1, x_test1, y_train1, y_test1 = train_test_split(df, Target_1, test_size=0.2, random_state=0)
print("Train data has {} data points, test data has {} data points" .format(x_train1.shape[0], x_test1.shape[0]))

# Target 2:
x_train2, x_test2, y_train2, y_test2 = train_test_split(df, Target_2, test_size=0.2, random_state=0)
print("Train data has {} data points, test data has {} data points" .format(x_train2.shape[0], x_test2.shape[0]))


# CREATE THE NEURAL NETWORK:

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(df))

# Model architecture:
model = keras.Sequential([
    normalizer,
    layers.Dense(8, input_dim=8),
    layers.Dense(4),
    layers.Dense(2),
    layers.Dense(1),
])

# Compile the model:
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='mae',
    metrics=['mean_absolute_error']
)
model.build()
model.summary()


# Train the model(s):
num_epochs = 30
batch_size = 150

history_1 = model.fit(x_train1, y_train1, epochs=num_epochs, validation_split=0.2)
history_2 = model.fit(x_train2, y_train2, epochs=num_epochs, validation_split=0.2)


# Evaluate the model:
loss1, mae1, mse1 = model.evaluate(x_test1, y_test1, verbose=2)
print(mae1)

loss2, mae2, mse2 = model.evaluate(x_test2, y_test2, verbose=2)
print(mae2)


# ASSESS THE PERFORMANCE OF THE ALGORITHM: Visualizing its accuracy and loss rate over each epoch will give us
# insight into whether or not the model is over/under fitting the data:


# Summarize history for loss: Heating load predictions
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for accuracy
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss: Cooling load predictions
plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Make the predictions:
Heating_load_predictions = model.predict(x_test1)
Cooling_load_predictions = model.predict(x_test2)


# Plot the predictions:

plt.scatter(y_test1, Heating_load_predictions)
plt.xlabel('True Heating Load')
plt.ylabel('Predicted Heating Load')
lims = [0, 60]
plt.xlim(lims)
plt.ylim(lims)

plt.scatter(y_test2, Cooling_load_predictions)
plt.xlabel('True Cooling Load')
plt.ylabel('Predicted Cooling Load')
lims = [0, 60]
plt.xlim(lims)
plt.ylim(lims)


# CONCLUSION: This sequential neural network accurately estimates the heating and cooling loads for commercial
# buildings based off of 8 different parameters. This model does not appear to be over fit, and will likely succeed in
# predicting heating/cooling loads for other buildings if the above parameters can be determined. It has a mean
# absolute error for the heating and cooling loads of 3.3 and 2.3 KiloWatts, respectively.


