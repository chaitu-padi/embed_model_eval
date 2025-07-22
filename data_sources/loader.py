import pandas as pd
from sqlalchemy import create_engine


def prepare_texts_for_embedding(df, embed_columns):
    """
    Concatenate selected columns as 'ColumnName: value' pairs for semantic embedding.
    """
    if embed_columns == 'all':
        embed_columns = df.columns.tolist()
    texts = []
    for _, row in df.iterrows():
        pairs = [f"{col}: {row[col]}" for col in embed_columns if col in df.columns]
        text = "; ".join(pairs)
        texts.append(text)
    return texts

def load_data(config):
    ds = config['data_source']
    texts = None
    if ds['type'] == 'oracle':
        engine = create_engine(f"oracle+cx_oracle://{ds['user']}:{ds['password']}@{ds['host']}:{ds['port']}/{ds['sid']}")
        df = pd.read_sql(f"SELECT * FROM {ds['table']}", engine)
        collection_name = ds['table']
    elif ds['type'] in ['csv', 'json', 'txt']:
        if ds['type'] == 'csv':
            df = pd.read_csv(ds['file'])
            embed_columns = ds.get('embed_columns', None)
            if embed_columns:
                texts = prepare_texts_for_embedding(df, embed_columns)
        elif ds['type'] == 'json':
            df = pd.read_json(ds['file'])
        elif ds['type'] == 'txt':
            df = pd.read_csv(ds['file'], delimiter=config.get('chunking', {}).get('delimiter', '\n'), header=None)
        collection_name = ds['file'].split('.')[0]
    else:
        raise ValueError(f"Unsupported data source type: {ds['type']}")
    return df, collection_name, texts
