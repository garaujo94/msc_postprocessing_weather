import requests
import pandas as pd
import json
from tqdm import tqdm


print('Starting...')

print('Getting API Credentials...')
f = open('credentials.json', )
credentials = json.load(f)
f.close()
print('Getting API Credentials...OK')
print('----')

print('Request available stations...')

# Frost API
endpoint = 'https://frost.met.no/observations/availableTimeSeries/v0.jsonld'
parameters = {
    'referencetime': '2018-01-01/2018-02-01',  # Change here to define initial date
    'elements': 'air_temperature'
}
# Issue an HTTP GET request
r = requests.get(endpoint, parameters, auth=(credentials['client_ID'], ''))
# Extract JSON data
data = r.json()
print('Request available stations...OK')
print('----')

# Only active stations
print('Filtering active stations...')
active_stations = []
for row in data['data']:
    if 'validTo' not in row.keys():
        active_stations.append(row)
print('Filtering active stations...OK')
print('----')

# Get only IDs
station_id = []
for station in active_stations:
    station_id.append(station['sourceId'])

# Get latlong
# URL is too big, it is necessary split in a few requests
print('Preparing URI to request latlong metadata...')
query_of_id = ''
querys = []
for station in station_id:
    query_of_id += f'{station.split(":")[0]}, '
    if len(query_of_id) > 1500:
        query_of_id = query_of_id[:-2]
        querys.append(query_of_id)
        query_of_id = ''
print('Preparing URI to request latlong metadata...OK')
print('----')

print('Request latlong metadata...')
# requests
id_lat_long = []
aux = {}
i = 0
session = requests.Session()
for query in tqdm(querys):
    # Define endpoint and parameters
    endpoint = 'https://frost.met.no/sources/v0.jsonld'
    parameters = {
        'ids': query
    }
    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(credentials['client_ID'], ''))
    # Extract JSON data
    # print(r)
    data = r.json()

    aux[i] = data['data']
    i += 1
print('Request latlong metadata...OK')
print('----')

id_lat_long = []
for a in aux:
    id_lat_long += aux[a]

print('Extracting latlong metadata...')
# grep latlong
complete_data = []
a = 0
for row in id_lat_long:
    answer = {}
    if 'geometry' in row:
        answer['station_id'] = row['id']

        answer['lat'] = row['geometry']['coordinates'][1]
        answer['long'] = row['geometry']['coordinates'][0]

    else:
        a += 1
        continue
    complete_data.append(answer)
print('Extracting latlong metadata...OK')
print('----')

# Missing values
print(f'Missing latlong values: {a}')
print(f'Total number of stations: {len(complete_data)}')

# Save csv file
print('Saving file as stations_id_locs.csv...')
pd.DataFrame(complete_data).to_csv('stations/stations_id_locs.csv', index=False)
print('Saving file as stations_id_locs.csv...OK')
