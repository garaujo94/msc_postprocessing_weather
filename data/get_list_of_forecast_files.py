import requests
from bs4 import BeautifulSoup
import json
import numpy as np
from tqdm import tqdm
import re
import argparse


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def scrap_url(url, years, months):
    complete_urls = []
    dates = []
    session = requests.Session()
    for year in tqdm(years, desc='Year'):
        for month in tqdm(months, desc='Month', leave=False):
            if len(str(month)) == 1:
                month = str(0) + str(month)
            url_by_year_month = f"{url}/{year}/{month}/catalog.html"

            response = session.get(url_by_year_month)
            soup = BeautifulSoup(response.text, 'html.parser')

            topics = soup.findAll('tt')
            days = []
            for i in topics:
                if len(i.get_text().strip()) == 3:
                    days.append(i.get_text().strip())
            for day in days:
                full_url = f"{url}/{year}/{month}/{day}catalog.html"
                complete_urls.append(full_url)

                answer = {
                    'year': year,
                    'month': month,
                    'day': day,
                    'url': f"{url}/{year}/{month}/{day}catalog.html"
                }
                dates.append(answer)
    return dates


def scrap_data(url_data, dates, match):
    full_data = []
    session = requests.Session()
    for row in tqdm(dates, desc='Pages/day'):
        # Extracting URL from dictionary
        url = row['url']

        # Getting page
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Getting tag of interest
        topics = soup.findAll('tt')

        # Iterate over results
        for topic in topics:
            content = topic.get_text().strip()
            m = re.search(match, content)
            if m:
                answer = {
                    'year': row['year'],
                    'month': row['month'],
                    'day': row['day'],
                    'hour': content[-6:-3],
                    'url_page': row['url'],
                    'file': content,
                    'uri': f'{url_data}/{row["year"]}/{row["month"]}/{row["day"]}{content}'
                }
                full_data.append(answer)
    return full_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='get 2.5km grid data', action='store_true')
    parser.add_argument('-o', help='get 1km grid data', action='store_true')
    parser.add_argument('--year_init', help='Start year')
    parser.add_argument('--year_final', help='End year')
    args = parser.parse_args()
    # Year to scrap

    years = np.arange(int(args.year_init), int(args.year_final) + 1).tolist()
    print(f'Year to look: {years}')

    # All months
    months = np.arange(1, 13)

    if args.t:
        print('=== MEPS 2.5km ===')
        print('Scrapping URLs...')
        url_to_search = 'https://thredds.met.no/thredds/catalog/meps25epsarchive'
        url_data = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive'

        datas = scrap_url(url_to_search, years, months)

        print('Scrapping URLs...OK')
        print('------')

        match = '^meps.*subset_2_5km.*'
        print('Scrapping URL files...')
        full_data = scrap_data(url_data, datas, match)
        print('Scrapping URL files...OK')
        print('------')

        # Your codes ....
        print('Start to save...')
        json.dump(full_data, open('crawler_data/crawler_data_2_5_km.json', 'w'), cls=NpEncoder)
        print('Saved as crawler_data_2_5_km.json')
    if args.o:
        print('=== GRIDPP 1km ===')
        print('Scrapping URLs...')

        url_to_search = 'https://thredds.met.no/thredds/catalog/metpparchive'
        url_data = 'https://thredds.met.no/thredds/dodsC/metpparchive'
        datas = scrap_url(url_to_search, years, months)

        print('Scrapping URLs...OK')
        print('------')

        match = r"(analysis.*)(18Z|12Z|06Z|00Z)"
        print('Scrapping URL files...')
        full_data = scrap_data(url_data, datas, match)
        print('Scrapping URL files...OK')
        print('------')

        # Your codes ....
        print('Start to save...')
        json.dump(full_data, open('crawler_data/crawler_data_1_km.json', 'w'), cls=NpEncoder)
        print('Saved as crawler_data_1_km.json')


if __name__ == '__main__':
    main()
