# This file includes a function that imports data from WorldWeatherOnline and translates it from a json file into a
# csv file.

# Imports
from wwo_hist import retrieve_hist_data

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
