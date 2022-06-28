# SOLAR ENERGY FORECASTS ACROSS NORTH AMERICA


# This project contains code that aims to accurately predict energy output (in Watts) in solar panels from a series
# of associated variables (i.e., latitude, pressure, humidity etc.) using neural network algorithms. Most
# importantly, it does not use solar irradiance as a predictor variable, due to the inconsistent and often inaccurate
# reporting of solar irradiance. The main idea behind this project is to forecast solar energy potential using
# commonly reported data across North America. Therefore, these models can be applied to a large amount of different
# locations to estimate solar panel performance.


# Import all required packages:
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes
import matplotlib.pyplot as plt
import mpl_scatter_density  # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from scipy.stats import kde
from tensorflow.keras import datasets, layers, models
from wwo_hist import retrieve_hist_data

pd.options.mode.chained_assignment = None  # default='warn'


# ----------------------------------------------------------------------------------------------------------------------

#  DATA PREPROCESSING, INSPECTION, AND FEATURE SELECTION
# Two sets of functions: 1) Data preprocessing 2) Feature selection and scoring
# These functions will act to prepare all the data before application of machine learning models:


# Create a function to do all of the data cleaning, with the required data frame as input:


def data_clean(input_csv):
    # Import the required dataset:
    data = pd.read_csv(input_csv)

    # Our target variable is solar output, referred to here as 'PolyPwr'.
    solar_power = data.pop("PolyPwr")  # Isolate solar output as an independent variable
    input_data = data  # Rename

    # Remove useless parameters
    df_updated = input_data.drop(['YRMODAHRMI', 'Location'], axis=1)

    # Create cyclic month features:
    df_updated['sine_mon'] = np.sin((df_updated.Month - 1) * np.pi / 11)
    df_updated['cos_mon'] = np.cos((df_updated.Month - 1) * np.pi / 11)

    # One hot encode 'Season':
    df_updated = pd.get_dummies(df_updated, columns=['Season'], drop_first=True)  # Season

    # Encode the datetime:
    df_updated['Date_raw'] = pd.to_datetime(df_updated['Date']).astype(np.int64)

    # Drop month and hour from our dataframe, as well as other redundant variables:
    cleaned_data = df_updated.drop(['Hour', 'Altitude', 'Cloud.Ceiling', 'Time', 'Date_raw', 'Date', 'Month'], axis=1)

    return cleaned_data, solar_power


# Create a function to rank all features by measure of influence, and eliminate any variables that are not well
# correlated with the target variable. in theory, this should optimize the machine learning algorithms.


def feature_selection(features, target):
    # Feature extraction
    selector = SelectKBest(score_func=f_classif, k='all').fit(features, target)
    scores = selector.scores_  # We now have a series of scores for each feature

    # Feature names:
    feature_names = list(features)

    # Order the variables by feature importance:
    feature_imp = pd.Series(scores, index=feature_names).sort_values(ascending=False)

    fig1 = plt.figure(figsize=(10, 9))
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()

    return fig1


# Apply the preprocessing function to the data file:
data_output = data_clean("Solar_Energy_Forecasting/Pasion et al dataset.csv")

df_variables = data_output[0]  # Predictor variables
Solar_power = data_output[1]  # Target variable: Solar power

# We only have latitude, time measurements, and a series of environmental variables now. These are the necessary
# inputs for this project.


# Feature selection is the next important step. It will allow us to identify the most influential variables in the
# dataset, and eliminate any variables that are of limited importance and may reduce model accuracy.
# Feature selection will be done using two Univariate selection:

# Visualize feature importance by applying the function:
feature_selection(df_variables, Solar_power)

# ----------------------------------------------------------------------------------------------------------------------

# CREATE THE MODELS:

# CREATING THE NEURAL NETWORK:

# Need to split the data into training and testing sets to build and test the model:
x_train1, x_test1, y_train1, y_test1 = train_test_split(df_variables, Solar_power, test_size=0.2, random_state=0)
print("Train data has {} data points, test data has {} data points".format(x_train1.shape[0], x_test1.shape[0]))


# Build a function that creates a deep neural network, with the training sets as input:


def neural_network(xtrain, ytrain, xtest, ytest, variables, activation_fn1, activation_fn2, learning_rate, loss_metric,
                   num_epochs, batch_size):

    # CREATE THE NEURAL NETWORK:
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(variables))

    # Model architecture:
    model = keras.Sequential([
        normalizer,
        layers.Dense(320, input_dim=xtrain.shape[1], activation=activation_fn1),
        layers.Dropout(0.2),
        layers.Dense(280, activation=activation_fn1),
        layers.Dropout(0.2),
        layers.Dense(240, activation=activation_fn2),
        layers.Dropout(0.2),
        layers.Dense(120, activation=activation_fn2),
        layers.Dropout(0.2),
        layers.Dense(60, activation=activation_fn2),
        layers.Dropout(0.2),
        layers.Dense(30, activation=activation_fn2),
        layers.Dense(1),
    ])

    # Model compilation:
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_metric,
        metrics=['mean_absolute_error']
    )

    model.build()

    # Train the DNN:
    num_epochs = num_epochs
    batch_size = batch_size
    history_1 = model.fit(xtrain, ytrain, epochs=num_epochs, validation_split=0.2)  # Fitting

    loss1, mae1 = model.evaluate(xtest, ytest, verbose=2)  # Calculate model mean absolute errors and loss rates

    return model, history_1


# Build a function that creates a deep Random Forest Algorithm, with the training sets as input:


def random_forest(xtrain, ytrain, n_estimators, random_state):

    # Instantiate model with x# of decision trees
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    rf.fit(xtrain, ytrain)  # Train the RF

    return rf


# Apply the DNN function:
DNN_output = neural_network(x_train1, y_train1, x_test1, y_test1, df_variables, 'relu', 'tanh', 0.001, 'mae', 40, 4000)

# ASSESS THE PERFORMANCE OF THE ALGORITHM: Visualizing its accuracy and loss rate over each epoch will give us
# insight into whether or not the model is over/under fitting the data:
DNN_model = DNN_output[0]  # Specify the DNN model
History = DNN_output[1]  # Specify the loss history

# Summarize history for loss: (This is only applicable for the DNN)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Apply the RF function:
RF_model = random_forest(x_train1, y_train1, 1000, 42)


# ----------------------------------------------------------------------------------------------------------------------

# DERIVE THE PREDICTIONS:

# DNN:
Solar_predictions_DNN = DNN_model.predict(x_test1)

# RF:
Solar_predictions_RF = RF_model.predict(x_test1)

# Evaluate the models accuracies:

# Mean absolute error between predictions and actual values:
RF_MAE = mae(y_test1, Solar_predictions_RF)
DNN_MAE = mae(y_test1, Solar_predictions_DNN)

# R2 score:
R2_DNN = r2_score(y_test1.ravel(), Solar_predictions_DNN)
R2_RF = r2_score(y_test1.ravel(), Solar_predictions_RF)


# Calculate the point density
xy_DNN = np.vstack([y_test1, Solar_predictions_DNN.reshape(-1)])
z_DNN = kde.gaussian_kde(xy_DNN)(xy_DNN)

xy_RF = np.vstack([y_test1, Solar_predictions_RF.reshape(-1)])
z_RF = kde.gaussian_kde(xy_RF)(xy_RF)

fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(y_test1, Solar_predictions_DNN, c=z_DNN, alpha=0.1)
ax1.set_title("Deep Neural Network")
ax1.set_ylabel("Predicted output (Watts/15min)")
ax1.set_xlabel("Measured output (Watts/15min)")
ax1.annotate('Mean absolute error: {}'.format(round(DNN_MAE, 2)), xy=(2, 33))
ax1.annotate('R squared: {}'.format(round(R2_DNN, 2)), xy=(2, 31))
lims = [0, 35]
plt.xlim(lims)
plt.ylim(lims)
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(y_test1, Solar_predictions_RF, c=z_RF, alpha=0.1)
ax2.set_title("Random Forest")
ax2.set_xlabel("Measured output (Watts/15min)")
plt.xlim(lims)
plt.ylim(lims)


# The Random forest algorithm yields the most accurate results on test data. Therefore we will use this model for the
# application of solar energy predictions.

#######################################################################################################################
