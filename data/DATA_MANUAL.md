# Data Manual of Usage

## get_list_of_forecast_files.py
```
usage: get_list_of_forecast_files.py [-h] [-t] [-o] [--year_init YEAR_INIT] [--year_final YEAR_FINAL]


optional arguments:
  -h, --help            show this help message and exit
  -t                    get 2.5km grid data
  -o                    get 1km grid data
  --year_init YEAR_INIT
                        Start year
  --year_final YEAR_FINAL
                        End year
                        
HINT: python get_list_of_forecast_files.py -t -o --year_init 2019 --year_final 2019
```
The Hint above will scrap links of 2019 of MEPS and GRIDPP.
```
Output: JSON File
Directory: /crawler_data/
```

## get_stations.py
```
python get_stations.py
```
It is needed to register on Frost API and create a file called "credentials.json" at the same folder of the python 
file with the shape
```
{
    "client_ID": "00000",
    "secret": "00000"
}
```
## get_all_forecasts.py
```
Soon...
```

## get_all_observation.py
```
Soon...
```

## ETL
```
Soon...
```