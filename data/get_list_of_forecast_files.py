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


def scrap_url(url, year, month):
    complete_urls = []
    dates = []
    for ano in tqdm(year, desc='Ano'):
        for mes in tqdm(month, desc='Mês', leave=False):
            if len(str(mes)) == 1:
                mes = str(0) + str(mes)
            url_por_ano_mes = f"{url}/{ano}/{mes}/catalog.html"
            response = requests.get(url_por_ano_mes)
            soup = BeautifulSoup(response.text, 'html.parser')

            topics = soup.findAll('tt')
            days = []
            for i in topics:
                if len(i.get_text().strip()) == 3:
                    days.append(i.get_text().strip())
            for day in days:
                url_completa = f"{url}/{ano}/{mes}/{day}catalog.html"
                complete_urls.append(url_completa)

                dicionario = {
                    'ano': ano,
                    'mes': mes,
                    'dia': day,
                    'url': f"{url}/{ano}/{mes}/{day}catalog.html"
                }
                dates.append(dicionario)
    return dates


def scrap_data(url_data, datas, match):
    full_data = []
    for row in tqdm(datas, desc='Pages/day'):
        # Extraindo URL do dicinário
        url = row['url']

        # Pegando página
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Pegando tag de interesse
        topics = soup.findAll('tt')

        # Iterando respostas
        for topic in topics:
            conteudo = topic.get_text().strip()
            m = re.search(match, conteudo)
            if m:
                dicionario = {
                    'ano': row['ano'],
                    'mes': row['mes'],
                    'dia': row['dia'],
                    'hora': conteudo[-6:-3],
                    'url_pagina_inicial': row['url'],
                    'arquivo': conteudo,
                    'uri': f'{url_data}/{row["ano"]}/{row["mes"]}/{row["dia"]}{conteudo}'
                }
                full_data.append(dicionario)
    return full_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='get 2.5km grid data', action='store_true')
    parser.add_argument('-o', help='get 1km grid data', action='store_true')
    parser.add_argument('--year_init', help='Start year')
    parser.add_argument('--year_final', help='End year')
    args = parser.parse_args()
    # Year to scrap

    anos = np.arange(args.year_init, args.year_final + 1).tolist()
    print(f'Year to look: {anos}')

    # All months
    meses = np.arange(1, 13)

    if args.t:
        print('=== MEPS 2.5km ===')
        print('Scrapping URLs...')
        url_to_search = 'https://thredds.met.no/thredds/catalog/meps25epsarchive'
        url_data = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive'

        datas = scrap_url(url_to_search, anos, meses)

        print('Scrapping URLs...OK')
        print('------')

        match = '^meps.*subset_2_5km.*'
        print('Scrapping URL files...')
        full_data = scrap_data(url_data, datas, match)
        print('Scrapping URL files...OK')
        print('------')

        # Your codes ....
        print('Start to save...')
        json.dump(full_data, open('crawler_data_2_5_km.json', 'w'), cls=NpEncoder)
        print('Saved as crawler_data_2_5_km.json')
    if args.o:
        print('=== GRIDPP 1km ===')
        print('Scrapping URLs...')

        url_to_search = 'https://thredds.met.no/thredds/catalog/metpparchive'
        url_data = 'https://thredds.met.no/thredds/dodsC/metpparchive'
        datas = scrap_url(url_to_search, anos, meses)

        print('Scrapping URLs...OK')
        print('------')

        match = r"(analysis.*)(18Z|12Z|06Z|00Z)"
        print('Scrapping URL files...')
        full_data = scrap_data(url_data, datas, match)
        print('Scrapping URL files...OK')
        print('------')

        # Your codes ....
        print('Start to save...')
        json.dump(full_data, open('crawler_data_1_km.json', 'w'), cls=NpEncoder)
        print('Saved as crawler_data_1_km.json')


if __name__ == '__main__':
    main()