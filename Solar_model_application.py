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
import os

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

# First, import the module 'Solar_model_creation_and_validation':
import Solar_Energy_Forecasting.Solar_model_creation_and_validation

# Define the Random Forest Algorithm created and validated in that module for application here:
rf = Solar_Energy_Forecasting.Solar_model_creation_and_validation.RF_model


# Input new data from a variety of sources, clean and preprocess:

frequency = 24
start_date = '01-JAN-2021'
end_date = '31-DEC-2021'
api_key = 'bc45c0b9b5794e198c1210622220304'
location_list = ['aberdeen', 'olympia', 'richland']

hist_weather_data = retrieve_hist_data(api_key,  # This uses a specific function available on github
                                       location_list,
                                       start_date,
                                       end_date,
                                       frequency,
                                       location_label=True,
                                       export_csv=True,
                                       store_df=True)

# Import the csv's created above as a data frame:
Salem = pd.read_csv("Solar_Energy_Forecasting/salem.csv")
Portland = pd.read_csv("Solar_Energy_Forecasting/portland.csv")
Yakima = pd.read_csv("Solar_Energy_Forecasting/yakima.csv")
Tacoma = pd.read_csv("Solar_Energy_Forecasting/tacoma.csv")
Seattle = pd.read_csv("Solar_Energy_Forecasting/seattle.csv")
Victoria = pd.read_csv("Solar_Energy_Forecasting/victoria.csv")
Vancouver = pd.read_csv("Solar_Energy_Forecasting/vancouver.csv")
Nanaimo = pd.read_csv("Solar_Energy_Forecasting/nanaimo.csv")
Kelowna = pd.read_csv("Solar_Energy_Forecasting/kelowna.csv")
Kamloops = pd.read_csv("Solar_Energy_Forecasting/kamloops.csv")
Aberdeen = pd.read_csv("Solar_Energy_Forecasting/aberdeen.csv")
Olympia = pd.read_csv("Solar_Energy_Forecasting/olympia.csv")
Richland = pd.read_csv("Solar_Energy_Forecasting/richland.csv")


# Design a function to perform data preprocessing to create a suitable dataset for the RF algorithm:
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


# Apply the function and prepare all the data. Create solar predictions using the RF model. Must multiply all
# predictions by 96 to get full day output, as eachprediciton is based on a 15min interval.

Salem_processed = ml_preprocess(Salem, 44.9429, -123.0351)
Salem_solar = (np.reshape(rf.predict(Salem_processed), (-1, 1))) * 96

Portland_processed = ml_preprocess(Portland, 45.5152, -122.6784)
Portland_solar = np.reshape(rf.predict(Portland_processed), (-1, 1)) * 96

Yakima_processed = ml_preprocess(Yakima, 46.602070, -120.505898)
Yakima_solar = np.reshape(rf.predict(Yakima_processed), (-1, 1)) * 96

Tacoma_processed = ml_preprocess(Tacoma, 47.2529, -122.4443)
Tacoma_solar = np.reshape(rf.predict(Tacoma_processed), (-1, 1)) * 96

Seattle_processed = ml_preprocess(Seattle, 47.6062, -122.3321)
Seattle_solar = np.reshape(rf.predict(Seattle_processed), (-1, 1)) * 96

Victoria_processed = ml_preprocess(Victoria, 48.4284, -123.3656)
Victoria_solar = np.reshape(rf.predict(Victoria_processed), (-1, 1)) * 96

Vancouver_processed = ml_preprocess(Vancouver, 49.2827, -123.1207)
Vancouver_solar = np.reshape(rf.predict(Vancouver_processed), (-1, 1)) * 96

Nanaimo_processed = ml_preprocess(Nanaimo, 49.1659, -123.9401)
Nanaimo_solar = np.reshape(rf.predict(Nanaimo_processed), (-1, 1)) * 96

Kelowna_processed = ml_preprocess(Kelowna, 49.8880, -119.4960)
Kelowna_solar = np.reshape(rf.predict(Kelowna_processed), (-1, 1)) * 96

Kamloops_processed = ml_preprocess(Kamloops, 50.6745, -120.3273)
Kamloops_solar = np.reshape(rf.predict(Kamloops_processed), (-1, 1)) * 96

Olympia_processed = ml_preprocess(Olympia, 47.037872, -122.900696)
Olympia_solar = np.reshape(rf.predict(Olympia_processed), (-1, 1)) * 96

Richland_processed = ml_preprocess(Richland, 44.7690, -117.1685)
Richland_solar = np.reshape(rf.predict(Richland_processed), (-1, 1)) * 96


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
Salem_df = avg_location(Salem_solar, Salem_processed)
Portland_df = avg_location(Portland_solar, Portland_processed)
Yakima_df = avg_location(Yakima_solar, Yakima_processed)
Tacoma_df = avg_location(Tacoma_solar, Tacoma_processed)
Seattle_df = avg_location(Seattle_solar, Seattle_processed)
Victoria_df = avg_location(Victoria_solar, Victoria_processed)
Vancouver_df = avg_location(Vancouver_solar, Vancouver_processed)
Nanaimo_df = avg_location(Nanaimo_solar, Nanaimo_processed)
Kelowna_df = avg_location(Kelowna_solar, Kelowna_processed)
Kamloops_df = avg_location(Kamloops_solar, Kamloops_processed)
Olympia_df = avg_location(Olympia_solar, Kamloops_processed)
Richland_df = avg_location(Richland_solar, Richland_processed)

# Merge the dataframes:
x = pd.concat([Salem_df, Portland_df, Yakima_df, Tacoma_df, Seattle_df, Victoria_df,
               Vancouver_df, Nanaimo_df, Kelowna_df, Kamloops_df, Olympia_df, Richland_df], axis=0)

#######################################################################################################################


# Plot the average daily solar power predictions on a map:

# Set the max/min latitude/longitude boundaries of the map:
min_lat = 43
max_lat = 53
min_lon = -127
max_lon = -118

# Average output figure:
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
plt.title('Average daily solar power for 2021', loc='center')
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cartopy.feature.LAND, zorder=1, edgecolor='k', facecolor='white')
ax1.add_feature(cfeature.BORDERS, zorder=2)
# ax1.add_feature(cfeature.LAKES, zorder=2)
ax1.set_extent([min_lon, max_lon, min_lat, max_lat],
               crs=ccrs.PlateCarree())
cb_sp = plt.scatter(x["Longitude"], x["Latitude"],
                    c=x["Average_output"], s=x["Average_output"] ,
                    cmap='plasma', edgecolors='k', zorder=100)  # Create a colour bar
fig.colorbar(cb_sp, ax=[ax1], fraction=0.023, pad=0.04, location='right')

# Make a heat map calendar for solar output on each day of the year!

# Need to make a function that calculates day of the year for each solar output estimation/day and input into
# dataframe for multiple locations.

# Do it for each city - then combine all figures!

fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(421)
ax1 = sns.heatmap(Victoria_solar, cmap='viridis')
ax1.title.set_text('Victoria')
ax1.set_axis_off()

ax2 = fig.add_subplot(422)
ax2 = sns.heatmap(Vancouver_solar, cmap='viridis')
ax2.title.set_text('Vancouver')
ax2.set_axis_off()

ax3 = fig.add_subplot(423)
ax3 = sns.heatmap(Seattle_solar, cmap='viridis')
ax3.title.set_text('Seattle')
ax3.set_axis_off()

ax4 = fig.add_subplot(424)
ax4 = sns.heatmap(Portland_solar, cmap='viridis')
ax4.title.set_text('Portland')
ax4.set_axis_off()

ax5 = fig.add_subplot(425)
ax5 = sns.heatmap(Kamloops_solar, cmap='viridis')
ax5.title.set_text('Kamloops')
ax5.set_axis_off()

ax6 = fig.add_subplot(426)
ax6 = sns.heatmap(Kelowna_solar, cmap='viridis')
ax6.title.set_text('Kelowna')
ax6.set_axis_off()

ax7 = fig.add_subplot(427)
ax7 = sns.heatmap(Olympia_solar, cmap='viridis')
ax7.title.set_text('Olympia')
ax7.set_axis_off()

ax8 = fig.add_subplot(428)
ax8 = sns.heatmap(Salem_solar, cmap='viridis')
ax8.title.set_text('Salem')
ax8.set_axis_off()