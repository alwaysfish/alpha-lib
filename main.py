from alphalib.datasets import *

data = load_hedgefund_strategies()
print(data.head())

data = load_market_sectors()
print(data.head())

data = load_market_sectors(dataset='equally_weighted')
print(data.head())

data = load_market_sectors(num_sectors=49)
print(data.head())

data = load_market_sectors(dataset='equally_weighted', num_sectors=49)
print(data.head())
