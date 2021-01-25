# The script updates historical prices given a list of tickers. All tickers must be provided
# in file tickers.csv

import datetime
from datetime import date
import pandas as pd
from pandas_datareader.yahoo.daily import YahooDailyReader
from pandas_datareader._utils import RemoteDataError


DATA_DIR = './data'


def get_end_date():
    date_to = datetime.datetime.now()
    if date_to.hour < 18:
        date_to -= datetime.timedelta(days=1)

    return date_to.date()


def download_hist_prices(tickers):

    date_to = get_end_date()

    for t in tickers:

        try:
            prices = pd.read_csv(f'{DATA_DIR}/{t}.csv', parse_dates=True, index_col=0)

        except FileNotFoundError:
            date_from = date(2010, 1, 1)
            prices = None

        else:
            date_from = (prices.index.max() + datetime.timedelta(days=1)).date()

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
                new_prices = new_prices.astype({'Volume': 'int32'})

                if prices is not None:
                    prices = prices.append(new_prices)
                else:
                    prices = new_prices

                prices.to_csv(f'{DATA_DIR}/{t}.csv', float_format='%.6g')

                print(f'{t}: new prices added for period [{date_from}, {date_to}]')
        else:
            print(f'{t}: no new prices found')


def main():
    tickers = pd.read_csv('tickers.csv', header=None)
    tickers.columns = ['tickers']

    download_hist_prices(tickers['tickers'])


if __name__ == '__main__':
    main()
