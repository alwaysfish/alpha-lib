# The script updates historical prices given a list of tickers. All tickers must be provided
# in file tickers.csv

import os
import datetime
from datetime import date
import pandas as pd
from pandas_datareader.yahoo.daily import YahooDailyReader
from pandas_datareader._utils import RemoteDataError
import psycopg2 as pql


DB_HOST = 'localhost'
DB_NAME = 'investing'
DB_USER = 'postgres'
TEMP_FILE = '__temp_prices.csv'


def get_end_date():
    date_to = datetime.datetime.now()
    if date_to.hour < 18:
        date_to -= datetime.timedelta(days=1)

    return date_to.date()


def download_hist_prices(tickers):

    date_to = get_end_date()

    conn = pql.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password='postgres')
    cur = conn.cursor()

    for t in tickers:

        # find the latest date for which a price exists
        cur.execute(f"SELECT max(recorded_at) FROM prices WHERE ticker='{t}'")

        # fetch functions always return tuples, hence the index [0]
        date_from = cur.fetchone()[0]
        
        if date_from is None:
            date_from = date(2010, 1, 1)
        else:
            date_from = date_from + datetime.timedelta(days=1)

        if (date_from <= date_to):
            try:
                reader = YahooDailyReader(t, date_from, date_to)
                new_prices = reader.read()

            except RemoteDataError:
                print(f"{t}: remote error")

            except KeyError:
                # There is a bug in pandas-datareader that doesn't check if the returned dataframe is empty
                # This is a workaround
                print(f'{t}: no new prices found')

            else:
                new_prices.rename(columns={
                    'High': 'high',
                    'Low': 'low',
                    'Open': 'open',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'adj_close'}, inplace=True)
                new_prices.index.rename('recorded_at', inplace=True)
                new_prices = new_prices.astype({'volume': 'int32'})
                new_prices['ticker'] = t

                new_prices[['ticker', 'high', 'low', 'open', 'close', 'adj_close', 'volume']].to_csv(
                    TEMP_FILE, header=False, float_format='%g'
                )

                with open(TEMP_FILE) as temp_csv:
                    cur.copy_from(temp_csv, 'prices', sep=',')
                
                conn.commit()
                os.remove(TEMP_FILE)

                print(f'{t}: new prices added for period [{date_from}, {date_to}]')
        else:
            print(f'{t}: no new prices found')


def main():
    tickers = pd.read_csv('tickers.csv', header=None)
    tickers.columns = ['tickers']

    download_hist_prices(tickers['tickers'])


if __name__ == '__main__':
    main()
