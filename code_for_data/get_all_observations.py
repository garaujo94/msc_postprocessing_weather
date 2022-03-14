import requests
import pandas as pd
import numpy as np
import json
from tqdm import tqdm, tqdm_gui
import datetime
import time
import logging
import os

if not os.path.exists('logs'):
    os.mkdir('logs')

logging.basicConfig(filename=f'logs/get_all_observations_{datetime.datetime.now()}.log',
                    level=logging.INFO,
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')


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
logging.info('Starting...')

print('Getting API Credentials...')
f = open('credentials.json', )
credentials = json.load(f)
print('Getting API Credentials...OK')
print('----')

# Forecast files -> Get Timestamp
with open('crawler_data/crawler_data_2_5_km.json', 'r') as d:
    forecast_files = json.load(d)

forecast_files = pd.DataFrame(forecast_files)
year_range = np.arange(2021, forecast_files.year.max() + 1)


# Stations data -> get station ID
stations = pd.read_csv('stations/stations_id_locs.csv')
stations = stations[stations.lat < 72]
stations = stations[stations.lat > 0]

# Final response DataFrame
final_response = []

# Endpoint
endpoint = 'https://frost.met.no/observations/v0.jsonld'

station_id = stations.station_id  # Colocar aqui
# station_id = iter(station_id)
# stations = []
# for x, y, z in zip(station_id, station_id, station_id):
#     stations.append([x, y, z])
stations = station_id


for year in tqdm(year_range):
    part = 0
    count = 0
    logging.info(f'Starting year {year}')
    timestamp_query = f'{year}-01-01/{year+1}-01-01'
    for query in tqdm(stations):
        try:
            count += 1
            parameters = {
                'sources': query,
                'referencetime': timestamp_query,
                'elements': 'air_temperature',
                'levels': 2,
                'qualities': 0,  
                'fields': 'sourceId, referenceTime, value'
            }

            r = requests.get(endpoint, parameters, auth=(credentials['client_ID'], ''))
            # Extract JSON data
            data = r.json()

            final_response.extend(data['data'])
        except:
            logging.error(f'Error {timestamp_query} at {query}')
        time.sleep(1)

        if count == 200:
            df = pd.DataFrame(final_response)
            df.to_csv(f'observation/observation_{year}_{part}.csv', index=False)
            del df
            
            # json.dump(final_response, open(f'observation/observation_{year}_{part}.json', 'w'), cls=NpEncoder)
            # print(f'Saved observations_{year}.json...')
            final_response = []
            count = 0
            part += 1



    df = pd.DataFrame(final_response)
    df.to_csv(f'observation/observation_{year}_{part}.csv', index=False)
    
    # json.dump(final_response, open(f'observation/observation_{year}_{part}.json', 'w'), cls=NpEncoder)
    # print(f'Saved observations_{year}_{part}.json...')
    final_response = []

logging.info('Done!')
