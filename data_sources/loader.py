import os
import pandas as pd
from sqlalchemy import create_engine


def prepare_texts_for_embedding(df, embed_columns):
    """
    Prepare texts for embedding while preserving original columns for payload.
    Returns both the texts for embedding and payload dictionaries.
    """
    if embed_columns == 'all':
        embed_columns = df.columns.tolist()
    
    texts = []
    payloads = []
    
    for _, row in df.iterrows():
        # Create the concatenated text for embedding
        pairs = [f"{col}: {row[col]}" for col in embed_columns if col in df.columns]
        text = "; ".join(pairs)
        texts.append(text)
        
        # Create payload dictionary with original values
        payload = {col: row[col] for col in embed_columns if col in df.columns}
        payloads.append(payload)
    
    return texts, payloads

def load_data(config):
    """Load data and return DataFrame, collection name, texts and payloads"""
    ds = config['data_source']
    
    # Use the collection name from vector_db config instead of file path
    collection_name = config['vector_db'].get('collection', 'flight_embeddings')
    
    if ds['type'] == 'csv':
        file_path = ds['file']
        if not os.path.isabs(file_path):
            file_path = os.path.join(
                os.path.dirname(__file__), 
                '../data/output', 
                os.path.basename(file_path)
            )
        df = pd.read_csv(file_path)
        
        # Prepare texts and payloads
        embed_columns = ds.get('embed_columns', df.columns.tolist())
        texts, payloads = prepare_texts_for_embedding(df, embed_columns)
        
        return df, collection_name, texts, payloads
    
    elif ds['type'] == 'oracle':
        engine = create_engine(f"oracle+cx_oracle://{ds['user']}:{ds['password']}@{ds['host']}:{ds['port']}/{ds['sid']}")
        df = pd.read_sql(f"SELECT * FROM {ds['table']}", engine)
        collection_name = ds['table']
        # If embed_columns is specified for oracle
        embed_columns = ds.get('embed_columns', None)
        if embed_columns:
            texts, payloads = prepare_texts_for_embedding(df, embed_columns)
    
    elif ds['type'] in ['json', 'txt']:
        if ds['type'] == 'json':
            df = pd.read_json(ds['file'])
            embed_columns = ds.get('embed_columns', None)
            if embed_columns:
                texts, payloads = prepare_texts_for_embedding(df, embed_columns)
        elif ds['type'] == 'txt':
            df = pd.read_csv(ds['file'], delimiter=config.get('chunking', {}).get('delimiter', '\n'), header=None)
            embed_columns = ds.get('embed_columns', None)
            if embed_columns:
                texts, payloads = prepare_texts_for_embedding(df, embed_columns)
        collection_name = ds['file'].split('.')[0]
    else:
        raise ValueError(f"Unsupported data source type: {ds['type']}")
    
    return df, collection_name, texts, payloads
