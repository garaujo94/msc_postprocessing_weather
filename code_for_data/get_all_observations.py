import requests
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import datetime
import time
import argparse
import os


if not os.path.exists('observation'):
    os.mkdir('observation')
    os.mkdir('observation/logs')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


print('Starting...')

print('Getting API Credentials...')
f = open('credentials.json', )
credentials = json.load(f)
print('Getting API Credentials...OK')
print('----')

# Forecast files -> Get Timestamp
with open('crawler_data/crawler_data_2_5_km.json', 'r') as d:
    forecast_files = json.load(d)

year = 2019
temp = []
for f in forecast_files:
    if f['year'] == year:
        temp.append(f)

forecast_files = temp

# Stations data -> get station ID
stations = pd.read_csv('stations/stations_id_locs.csv')
stations = stations[stations.lat < 72]
stations = stations[stations.lat > 0]

# Final response DataFrame
final_response = []

# Endpoint
endpoint = 'https://frost.met.no/observations/v0.jsonld'

station_id = stations.station_id  # Colocar aqui

query_of_id = ''
querys = []
for station in station_id:
    query_of_id += f'{station.split(":")[0]}, '
    if len(query_of_id) > 1500:
        query_of_id = query_of_id[:-2]
        querys.append(query_of_id)
        query_of_id = ''

log = open(f'observation/logs/log_{datetime.datetime.now()}.txt', 'w')
error_list = []
k = 0
part = 0
print(f'YEAR: {year}')
for row in tqdm(forecast_files):
    k += 1
    timestamp_query = f'{row["year"]}-{row["month"]}-{row["day"][:-1]}T{row["hour"][:-1]}:00:00'
    i = 0
    aux = {}
    for query in querys:
        try:
            parameters = {
                'sources': query,
                'referencetime': timestamp_query,
                'elements': 'air_temperature',
                'levels': 2,
                'qualities': 0  # Ainda não pesquisei assim
            }

            r = requests.get(endpoint, parameters, auth=(credentials['client_ID'], ''))
            # Extract JSON data
            data = r.json()

            aux[i] = data['data']
        except:
            log.write(f'Error {timestamp_query} at {query}\n')
        i += 1
        time.sleep(1)

    id_lat_long = []
    for a in aux:
        id_lat_long += aux[a]

    final_response += id_lat_long
    if k % 5 == 0:
        json.dump(final_response, open(f'observation/observation_{part}.json', 'w'), cls=NpEncoder)
        print(f'Saved observations_{part}.json...')
        final_response = []
        part += 1

    i += 1

# Your codes....
print('Start to save...')
json.dump(final_response, open(f'observation/observation_{part}.json', 'w'), cls=NpEncoder)
print('Saved as observation.json')
