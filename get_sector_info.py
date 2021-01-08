import pandas as pd


print('Downloading file')
df = pd.read_excel('https://nasdaqbaltic.com/statistics/en/shares?download=1')

if df is None:
    exit(1)

print('Saving data into CSV')
df.to_csv('./data/baltic_equities_info.csv', index=False, float_format='%.6g')

print('Done!')
