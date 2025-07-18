import pandas as pd
from sqlalchemy import create_engine

def load_data(config):
    ds = config['data_source']
    if ds['type'] == 'oracle':
        engine = create_engine(f"oracle+cx_oracle://{ds['user']}:{ds['password']}@{ds['host']}:{ds['port']}/{ds['sid']}")
        df = pd.read_sql(f"SELECT * FROM {ds['table']}", engine)
        collection_name = ds['table']
    elif ds['type'] in ['csv', 'json', 'txt']:
        if ds['type'] == 'csv':
            df = pd.read_csv(ds['file'])
        elif ds['type'] == 'json':
            df = pd.read_json(ds['file'])
        elif ds['type'] == 'txt':
            df = pd.read_csv(ds['file'], delimiter=config.get('chunking', {}).get('delimiter', '\n'), header=None)
        collection_name = ds['file'].split('.')[0]
    else:
        raise ValueError(f"Unsupported data source type: {ds['type']}")
    return df, collection_name
