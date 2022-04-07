# This file includes a function that imports data from WorldWeatherOnline and translates it from a json file into a
# csv file.

# Imports
from wwo_hist import retrieve_hist_data
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

# Specify parameters:
frequency = 24
start_date = '01-JAN-2021'
end_date = '05-JAN-2021'
api_key = 'bc45c0b9b5794e198c1210622220304'
location_list = ['toronto', 'vancouver', 'montreal', 'seattle', 'winnipeg']

hist_weather_data = retrieve_hist_data(api_key,
                                       location_list,
                                       start_date,
                                       end_date,
                                       frequency,
                                       location_label=True,
                                       export_csv=True,
                                       store_df=True)


# Import the csv's created above as a data frame:
Vancouver = pd.read_csv("vancouver.csv")


# Design a function to perform data preprocessing to create a suitable dataset for the neural network:

def ml_preprocess(data, latitude, longitude):

    # Select only the relevant data:
    df = data.iloc[:, [0, 17, 19, 20, 21, 23]]
    df.columns = ["Date", "Humidity", "Pressure", "AmbientTemp", "Visibility", "Wind.Speed"]

    # Input the coordinates for each city:
    df['Latitude'] = latitude
    df['Longitude'] = longitude

    # Date to datetime:
    df["Date"] = pd.to_datetime(df["Date"])

    # Create a month variable:
    df['Month'] = df['Date'].dt.month

    return df


x = ml_preprocess(Vancouver, 49.2827, -123.1207)


