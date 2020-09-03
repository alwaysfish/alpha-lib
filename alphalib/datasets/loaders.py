import pandas as pd
from os.path import dirname, join


def load_hedgefund_rets() -> pd.DataFrame:
    """
    Load and return monthly returns of 13 different hedge fund strategies.

    Returns:
        returns: pandas DataFrame
    """
    module_path = dirname(__file__)
    csv_filename = join(module_path, 'data', 'edhec-hedgefundindices.csv')

    data = pd.read_csv(csv_filename, index_col=0, parse_dates=True, dayfirst=True)
    data.index = data.index.to_period('M')
    data /= 100

    return data


def load_industry_rets(dataset='value_weighted', num_sectors=30) -> pd.DataFrame:
    """
    Load and return monthly returns of 30 or 49 different sectors. Value weighted or equally weighted returns
    are returned.

    Arguments:
        dataset: {'value_weighted', 'equally_weighted'}, default='value_weighted'

        num_sectors: {30, 49}, default=30
            Option to load and return either 30 or 49 different sectors.

    Returns:
        returns: pandas DataFrame
    """

    fname = ""

    if num_sectors == 30:
        if dataset == 'value_weighted':
            fname = 'ind30_m_vw_rets.csv'
        elif dataset == 'equally_weighted':
            fname = 'ind30_m_ew_rets.csv'
        else:
            raise ValueError("Parameter 'dataset' only accepts values 'value_weighted' or 'equally_weighted''")
    elif num_sectors == 49:
        if dataset == 'value_weighted':
            fname = 'ind49_m_vw_rets.csv'
        elif dataset == 'equally_weighted':
            fname = 'ind49_m_ew_rets.csv'
        else:
            raise ValueError("Parameter 'dataset' only accepts values 'value_weighted' or 'equally_weighted'")

    module_path = dirname(__file__)
    csv_filename = join(module_path, 'data', fname)

    data = pd.read_csv(csv_filename, index_col=0)

    # Some columns contain trailing spaces, therefore need to remove them
    data.columns = data.columns.str.strip()
    data.index = pd.to_datetime(data.index, format='%Y%m').to_period('M')
    data /= 100

    return data
