
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

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import r2_score
from scipy.stats import kde
from tensorflow.keras import datasets, layers, models
from wwo_hist import retrieve_hist_data

pd.options.mode.chained_assignment = None  # default='warn'


# Import the required dataset to train the neural network model:
data = pd.read_csv("Solar_Energy_Forecasting/Pasion et al dataset.csv")


# Our target variable is solar output, referred to here as 'PolyPwr'.
Solar_power = data.pop("PolyPwr")  # Isolate solar output as an independent variable
df = data  # Rename


#######################################################################################################################

#  DATA PREPROCESSING, INSPECTION, AND FEATURE SELECTION


# Now, we need to remove all the location specific data (i.e. specific region names) -  we want to build a model that
# can be extrapolated into new regions.
df_updated = df.drop(['YRMODAHRMI', 'Location'], axis=1)


# Create cyclic month features:
df_updated['sine_mon'] = np.sin((df_updated.Month - 1)*np.pi/11)
df_updated['cos_mon'] = np.cos((df_updated.Month - 1)*np.pi/11)


# One hot encode 'Season':
df_updated = pd.get_dummies(df_updated, columns=['Season'], drop_first=True)  # Season


# Encode the datetime:
df_updated['Date_raw'] = pd.to_datetime(df_updated['Date']).astype(np.int64)


# Now, we can drop month and hour from our dataframe, as well as other redundant variables:
df_variables = df_updated.drop(['Hour', 'Altitude', 'Cloud.Ceiling', 'Time', 'Date_raw', 'Date', 'Month'], axis=1)


# We only have latitude, time measurements, and a series of environmental variables now. These are the necessary
# inputs for this project.

# Feature selection is the next important step. It will allow us to identify the most influential variables in the
# dataset, and eliminate any variables that are of limited importance and may reduce model accuracy.
# Feature selection will be done using two Univariate selection:


# Univariate Selection:
X = df_variables
Y = Solar_power


# Feature extraction
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

#######################################################################################################################

# CREATING THE NEURAL NETWORK

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
    layers.Dropout(0.2),
    layers.Dense(280, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(240, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(120, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(60, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(30, activation='tanh'),
    layers.Dense(1),
])


# Model compilation:
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss='mae',
    metrics=['mean_absolute_error']
)
model.build()
model.summary()  # Inspect


# Train the model(s):
num_epochs = 40
batch_size = 4000
history_1 = model.fit(x_train1, y_train1, epochs=num_epochs, validation_split=0.2)  # Fitting


#######################################################################################################################

# EVALUATE PERFORMANCE:

# ASSESS THE PERFORMANCE OF THE ALGORITHM: Visualizing its accuracy and loss rate over each epoch will give us
# insight into whether or not the model is over/under fitting the data:


loss1, mae1 = model.evaluate(x_test1, y_test1, verbose=2)  # Calculate model mean absolute errors and losss rates
print(mae1)


# Summarize history for loss: Solar predictions
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# DERIVE THE PREDICTIONS:
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


# Visualize:
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='jet')
plt.show()


#######################################################################################################################

# APPLICATION:

# Input new data from a variety of sources, clean and preprocess:

frequency = 24
start_date = '01-AUG-2021'
end_date = '31-AUG-2021'
api_key = 'bc45c0b9b5794e198c1210622220304'
location_list = ['toronto', 'vancouver', 'houston', 'seattle', 'winnipeg']

hist_weather_data = retrieve_hist_data(api_key,  # This uses a specific function available on github
                                       location_list,
                                       start_date,
                                       end_date,
                                       frequency,
                                       location_label=True,
                                       export_csv=True,
                                       store_df=True)


# Import the csv's created above as a data frame:
Vancouver = pd.read_csv("vancouver.csv")
Toronto = pd.read_csv("toronto.csv")
Houston = pd.read_csv("houston.csv")


# Design a function to perform data preprocessing to create a suitable dataset for the neural network:
def ml_preprocess(data_csv, latitude, longitude):

    df = data_csv.iloc[:, [0, 17, 19, 20, 21, 23]]  # Select only the relevant columns
    df.columns = ["Date_raw", "Humidity", "Pressure", "AmbientTemp", "Visibility", "Wind.Speed"]  # Rename columns

    df['Latitude'] = latitude  # Input the latitude for each city
    df['Longitude'] = longitude  # Input the longitude for each city

    df["Date_raw"] = pd.to_datetime(df["Date_raw"])  # Date to datetime

    df['Month'] = df['Date_raw'].dt.month  # Create a month variable

    # Create cyclic month features
    df['sine_mon'] = np.sin((df.Month - 1) * np.pi / 11)
    df['cos_mon'] = np.cos((df.Month - 1) * np.pi / 11)

    # Date as an integer (i.e., 20210101)
    df['Date'] = (df['Date_raw'].dt.year * 10000 +
                  df['Date_raw'].dt.month * 100 +
                  df['Date_raw'].dt.day * 1)

    # Create season categories:
    df['Season'] = pd.cut(df.Month, bins=[0, 2, 5, 8, 11, 12],
                          labels=['Winter', 'Spring', 'Summer', 'Fall', 'Winter'],
                          ordered=False)

    # One hot encode season:
    df_updated = pd.get_dummies(df, columns=['Season'], drop_first=True)

    # Drop unnecessary variables:
    df_variables = df_updated.drop(['Date_raw', 'Date', 'Month'], axis=1)

    # Reorder columns:
    df_variables = df_variables[["Latitude", "Longitude", "Humidity", "AmbientTemp", "Wind.Speed",
                                 "Visibility", "Pressure", "sine_mon", "cos_mon", "Season_Spring",
                                 "Season_Summer", "Season_Winter"]]

    return df_variables


# Apply the function and prepare all the data:
Vancouver_processed = ml_preprocess(Vancouver, 49.2827, -123.1207)
Vancouver_solar = model.predict(Vancouver_processed)

Toronto_processed = ml_preprocess(Toronto, 43.6532, -70.3832)
Toronto_solar = model.predict(Toronto_processed)

Houston_processed = ml_preprocess(Houston, 29.7604, -95.3698)
Houston_solar = model.predict(Houston_processed)


#######################################################################################################################


# Create a function that creates a dataframe that averages total solar output (for the given time range) with lat and
# lon for each city:


def avg_location(solar_output, processed_data):

    avg_solar_output = pd.DataFrame([np.average(solar_output)])  # Average solar power as a dataframe

    cumulative_solar_output = pd.DataFrame([np.sum(solar_output)])  # Average solar power as a dataframe

    lat = pd.DataFrame([np.max(processed_data.Latitude)])  # Latitude as a dataframe, using np.max to isolate one row
    lon = pd.DataFrame([np.max(processed_data.Longitude)])  # Longitude as a dataframe, using np.max to isolate one row

    dataframe = pd.concat([avg_solar_output, cumulative_solar_output, lat, lon], axis=1)  # Merge
    dataframe.columns = ["Average_output", "Cumulative_output", "Latitude", "Longitude"]

    return dataframe


# Apply the average location function:
Houston_df = avg_location(Houston_solar, Houston_processed)
Toronto_df = avg_location(Toronto_solar, Toronto_processed)
Vancouver_df = avg_location(Vancouver_solar, Vancouver_processed)


# Merge the dataframes:
x = pd.concat([Houston_df, Vancouver_df, Toronto_df], axis=0)


#######################################################################################################################


# Plot the average monthly solar power predictions on a map:

# Set the max/min latitude/longitude boundaries of the map:
min_lat = 25
max_lat = 53
min_lon = -127
max_lon = -70


# Average output figure:
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
plt.title('Average monthly solar power', loc='center')
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='k', facecolor='white')
ax1.add_feature(cfeature.BORDERS, zorder=2)
# ax1.add_feature(cfeature.LAKES, zorder=2)
ax1.set_extent([min_lon, max_lon, min_lat, max_lat],
              crs=ccrs.PlateCarree())
cb_sp = plt.scatter(x["Longitude"], x["Latitude"],
                    c=x["Average_output"], s=x["Average_output"]*10,
                    cmap='plasma', edgecolors='k', zorder=100)  # Create a colour bar
fig.colorbar(cb_sp, ax=[ax1], fraction=0.023, pad=0.04, location='right')


# Cumulative output figure:
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
plt.title('Cumulative monthly solar power', loc='center')
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='k', facecolor='white')
ax1.add_feature(cfeature.BORDERS, zorder=2)
# ax1.add_feature(cfeature.LAKES, zorder=2)
ax1.set_extent([min_lon, max_lon, min_lat, max_lat],
              crs=ccrs.PlateCarree())
cb_sp = plt.scatter(x["Longitude"], x["Latitude"],
                    c=x["Cumulative_output"], s=x["Cumulative_output"],
                    cmap='plasma', edgecolors='k', zorder=100)  # Create a colour bar
fig.colorbar(cb_sp, ax=[ax1], fraction=0.023, pad=0.04, location='right')


# Make a heat map calendar for solar output on each day of the year!

# Need to make a function that caluclates day of the year for each solar output estimation/day and input into
# dataframe for multiple locations.


