import pandas as pd
import os
import re
from datetime import datetime

# Support Functions
def is_csv(x):
    if x[-4:] == '.csv':
        return True
    return False

def read_all_files(path, files):
    df = pd.DataFrame()
    for arquivo in files:
        df_aux = pd.read_csv(f'{path}/{arquivo}')
        df = df.append(df_aux)

    return df

def kelvin_to_celsius(k):
    return k - 273

def print_status(df):
    print(df.shape)
    print('='*10)
    print(df.info())

def str_to_datime(x):
    return datetime.strptime(x[:-5].replace('T', ' '), '%Y-%m-%d %H:%M:%S')


# Main
def main():
    forecast_path = 'forecasts/2.5'
    gridpp_path = 'forecasts/1'
    observation_path = 'observation'

    desired_columns = ['station_id', 'lat', 'long', 'forecast', 'gridpp', 'observations', 'year', 'month', 'day', 'hour']

    to_transform = ['year', 'month', 'day', 'hour'] # to int

    years = [2019, 2020, 2021] # change to argparse


    for year in years:
        print(f'=== YEAR {year} ===')

        print('Reading Forecasts...')
        forecast_files = os.listdir(forecast_path)
        forecast_files = list(filter(is_csv, forecast_files))

        forecasts = read_all_files(forecast_path, forecast_files)
        forecasts = forecasts[forecasts.year == year]

        forecasts.reset_index(drop=True, inplace=True)
        forecasts['forecast'] = forecasts['forecast'].apply(lambda x: kelvin_to_celsius(x))

        for item in to_transform:
            forecasts = forecasts.astype({item: 'int32'})

        forecasts['datetime'] = forecasts.apply(lambda x: datetime(x['year'], x['month'], x['day'], x['hour']), axis=1)
        print('Done')

        print('Reading Gridpp...')
        gridpp_files = os.listdir(gridpp_path)
        gridpp_files = list(filter(is_csv, gridpp_files))

        gridpp = read_all_files(gridpp_path, gridpp_files)
        gridpp = gridpp[gridpp.year == year]

        gridpp.reset_index(drop=True, inplace=True)
        gridpp['forecast'] = gridpp['forecast'].apply(lambda x: kelvin_to_celsius(x))

        for item in to_transform:
            gridpp = gridpp.astype({item: 'int32'})

        gridpp['datetime'] = gridpp.apply(lambda x: datetime(x['year'], x['month'], x['day'], x['hour']), axis=1)

        # Only for gridpp
        gridpp.rename(columns={'forecast': 'gridpp'}, inplace=True)
        print("Done")

        print('Reading Observations...')
        observation = pd.read_csv(f'{observation_path}/observation_{year}.csv')

        observation['observations'] = observation['observations'].apply(lambda x: x.split(':'))
        re_to_extract_numbers = r'\-*\d+\.*\d*'
        observation['observations'] = observation['observations'].apply(lambda x: float(re.findall(re_to_extract_numbers, x[-1])[0]))

        observation['datetime'] = observation.referenceTime.apply(lambda x: str_to_datime(x))

        observation['sourceId'] = observation['sourceId'].apply(lambda x: x.split(':')[0])
        print("Done")

        print("Merging...")
        final_data = forecasts.merge(gridpp[['station_id', 'datetime', 'gridpp']], how='inner', on=['station_id', 'datetime'], copy=False, suffixes=('_f', '_g'))
        final_data = final_data.merge(observation, how='inner', left_on=['station_id', 'datetime'], right_on=['sourceId', 'datetime'])
        final_data = final_data[desired_columns]
        print('Done')

        final_data.to_csv(f'../data/final_data_{year}.csv', index=False)


        del forecasts
        del gridpp
        del observation
        del final_data


if __name__ == '__main__':
    main()