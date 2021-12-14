import gridpp
import netCDF4
import json
import pandas as pd
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-t', help='get 2.5km grid data', action='store_true')
parser.add_argument('-o', help='get 1km grid data', action='store_true')
args = parser.parse_args()

# Stations data
stations = pd.read_csv('stations/stations_id_locs.csv')

if args.t:
    print('=== 2.5km data ===')
    # Final DataFrame
    final_response = pd.DataFrame(
        columns={'station_id', 'lat', 'long', 'indexes', 'forecast', 'nearest_forecast_lat', 'nearest_forecast_long'})
    # Forecast files
    with open('crawler_data/crawler_data_2_5_km.json', 'r') as d:
        forecast_files = json.load(d)

    print('Starting to take forecast data...')
    start = 0
    part = 0
    i = 1
    for row in tqdm(forecast_files):
        filename = row['uri']
        file = netCDF4.Dataset(filename)

        # Define latlong only once
        if start == 0:
            lats = file.variables['latitude'][:]
            long = file.variables['longitude'][:]
            grid = gridpp.Grid(lats, long)
            start += 1

            # Finding stations indexes on grid
            stations['indexes'] = stations[['lat', 'long']].apply(lambda x: grid.get_nearest_neighbour(x[0], x[1]),
                                                                  axis=1)
            stations['nearest_forecast_lat'] = stations['indexes'].apply(lambda x: lats[x[0], x[1]])
            stations['nearest_forecast_long'] = stations['indexes'].apply(lambda x: long[x[0], x[1]])

        # Extrating all temperatures and after get only for stations indexes
        # Although strange, this path proved to be absurdly faster
        temperature = file.variables['air_temperature_2m'][0, 0, 0, :, :]

        stations['forecast'] = stations['indexes'].apply(lambda x: temperature[x[0], x[1]])

        stations['year'] = row['year']
        stations['month'] = row['month']
        stations['day'] = row['day'][:-1]
        stations['hour'] = row['hour'][:-1]

        final_response = final_response.append(stations)

        if i % 200 == 0:
            final_response.to_csv(f'forecasts/forecasts_2_5_km_{part}.csv', index=False)
            print(f'Saved forecasts_2_5_km_{part}.csv...')
            final_response = pd.DataFrame(columns={'station_id',
                                                   'lat',
                                                   'long',
                                                   'indexes',
                                                   'forecast',
                                                   'nearest_forecast_lat',
                                                   'nearest_forecast_long'})
            part += 1

        i += 1

    final_response.to_csv(f'forecasts/forecasts_2_5_km_{part}.csv', index=False)

if args.o:
    print('=== 1km data ===')
    # Final DataFrame
    final_response = pd.DataFrame(
        columns={'station_id', 'lat', 'long', 'indexes', 'forecast', 'nearest_gridpp_lat', 'nearest_gridpp_long'})
    # Forecast files
    with open('crawler_data_1_km.json', 'r') as d:
        forecast_files = json.load(d)

    print('Starting to take forecast data...')
    start = 0
    part = 0
    i = 1
    for row in tqdm(forecast_files):
        filename = row['uri']
        file = netCDF4.Dataset(filename)

        # Define latlong only once
        if start == 0:
            lats = file.variables['latitude'][:]
            long = file.variables['longitude'][:]
            grid = gridpp.Grid(lats, long)
            start += 1

            # Finding stations indexes on grid
            stations['indexes'] = stations[['lat', 'long']].apply(lambda x: grid.get_nearest_neighbour(x[0], x[1]),
                                                                  axis=1)
            stations['nearest_gridpp_lat'] = stations['indexes'].apply(lambda x: lats[x[0], x[1]])
            stations['nearest_gridpp_long'] = stations['indexes'].apply(lambda x: long[x[0], x[1]])

        # Extrating all temperatures and after get only for stations indexes
        # Although strange, this path proved to be absurdly faster
        temperature = file.variables['air_temperature_2m'][0, :, :]

        stations['forecast'] = stations['indexes'].apply(lambda x: temperature[x[0], x[1]])

        stations['year'] = row['year']
        stations['month'] = row['month']
        stations['day'] = row['day'][:-1]
        stations['hour'] = row['hour'][:-1]

        final_response = final_response.append(stations)

        if i % 200 == 0:
            final_response.to_csv(f'forecasts/forecasts_1_km_{part}.csv', index=False)
            print(f'Saved forecasts_1_km_{part}.csv...')
            final_response = pd.DataFrame(columns={'station_id',
                                                   'lat',
                                                   'long',
                                                   'indexes',
                                                   'forecast',
                                                   'nearest_forecast_lat',
                                                   'nearest_forecast_long'})
            part += 1

        i += 1

    final_response.to_csv(f'forecasts/forecasts_1_km_{part}.csv', index=False)