import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from pymilvus import MilvusClient, connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import socket
# Add imports for other vector DBs
from config.config_manager import ConfigManager

def get_embedding_models():
    return [
        'all-MiniLM-L6-v2',
        'paraphrase-MiniLM-L3-v2',
        'multi-qa-MiniLM-L6-cos-v1',
        'openai-ada-002',
        'cohere-embed-english-v3',
        # Add more HuggingFace, OpenAI, Cohere, etc.
    ]

# Placeholder for chunking methods
def get_chunking_methods():
    return ['none', 'sentence', 'sliding_window', 'fixed_length']

# Connect to Milvus
def connect_milvus(host, port):
    try:
        connections.connect(host=host, port=port)
    except Exception as e:
        st.error(f"Failed to connect to Milvus at {host}:{port}. Error: {e}")
        raise

def ensure_milvus_collection(collection_name, dim):
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
    if collection_name in [c.name for c in Collection.list()]:
        return Collection(collection_name)
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Embedding collection")
    return Collection(collection_name, schema)

# UI for data source selection
def data_source_ui():
    st.sidebar.header('Data Source')
    source = st.sidebar.selectbox('Choose data source', ['Oracle', 'Data File'])
    if source == 'Oracle':
        host = st.sidebar.text_input('Oracle Host')
        port = st.sidebar.text_input('Oracle Port')
        sid = st.sidebar.text_input('Oracle SID')
        user = st.sidebar.text_input('Oracle User')
        password = st.sidebar.text_input('Oracle Password', type='password')
        table = st.sidebar.text_input('Table Name')
        return {'type': 'oracle', 'host': host, 'port': port, 'sid': sid, 'user': user, 'password': password, 'table': table}
    else:
        file = st.sidebar.file_uploader('Upload Data File', type=['csv', 'txt', 'json'])
        return {'type': 'file', 'file': file}

# UI for embedding configuration
def embedding_config_ui():
    st.sidebar.header('Embedding Configuration')
    model = st.sidebar.selectbox('Embedding Model', get_embedding_models())
    chunking = st.sidebar.selectbox('Chunking Method', get_chunking_methods())
    chunk_size = st.sidebar.number_input('Chunk Size', min_value=1, max_value=2048, value=128)
    parallelism = st.sidebar.number_input('Parallelism (Threads)', min_value=1, max_value=32, value=4)
    batch_size = st.sidebar.number_input('Batch Size', min_value=1, max_value=1024, value=64)
    dimension = st.sidebar.number_input('Embedding Dimension', min_value=1, max_value=4096, value=768)
    indexing_type = st.sidebar.selectbox('Indexing Type', ['flat', 'ivf', 'hnsw'])
    embedding_type = st.sidebar.selectbox('Embedding Type', ['semantic', 'dense', 'sparse'])
    search_type = st.sidebar.selectbox('Search Type', ['semantic', 'dense', 'sparse'])
    return {
        'model': model,
        'chunking_strategy': chunking,
        'chunk_size': chunk_size,
        'parallelism': parallelism,
        'batch_size': batch_size,
        'dimension': dimension,
        'indexing_type': indexing_type,
        'embedding_type': embedding_type,
        'search_type': search_type
    }

# UI for Milvus connection
def vector_db_config_ui():
    st.sidebar.header('Vector DB Configuration')
    db_type = st.sidebar.selectbox('Vector DB Type', ['Milvus', 'Qdrant'])
    host = st.sidebar.text_input(f'{db_type} Host', value='localhost')
    port = st.sidebar.text_input(f'{db_type} Port', value='19530' if db_type == 'Milvus' else '6333')
    collection = st.sidebar.text_input('Collection Name', value='embeddings')
    return {'db_type': db_type, 'host': host, 'port': port, 'collection': collection}

# Check if port is open
def is_port_open(host, port):
    try:
        with socket.create_connection((host, int(port)), timeout=3):
            return True
    except Exception:
        return False

# Main UI
def main():
    st.title('Embedding Performance Evaluation')
    config_manager = ConfigManager()
    data_source = data_source_ui()
    embed_config = embedding_config_ui()
    vector_db_config = vector_db_config_ui()

    # Right panel: show saved configurations
    st.sidebar.header('Saved Configurations')
    saved_configs = config_manager.list_configs()
    selected_config = st.sidebar.selectbox('Select Configuration', [''] + saved_configs)
    if selected_config:
        loaded = config_manager.load_config(selected_config)
        st.sidebar.write(f"Loaded config: {selected_config}")
        st.sidebar.json(loaded)

    # Option to save current config
    config_id = st.sidebar.text_input('Config ID to Save')
    if st.sidebar.button('Save Configuration'):
        config_manager.save_config({
            'data_source': data_source,
            'embed_config': embed_config,
            'vector_db_config': vector_db_config
        }, config_id)
        st.sidebar.success(f'Configuration {config_id} saved!')

    if st.button('Run Embedding & Evaluate'):
        import time
        st.info('Running embedding and evaluation...')
        db_type = vector_db_config['db_type']
        host = vector_db_config['host']
        port = vector_db_config['port']
        collection_name = vector_db_config['collection']
        if not is_port_open(host, port):
            st.error(f"Cannot connect to {db_type} at {host}:{port}. Please ensure the vector DB is running.")
            return
        # Load data
        if data_source['type'] == 'oracle':
            engine = create_engine(f"oracle+cx_oracle://{data_source['user']}:{data_source['password']}@{data_source['host']}:{data_source['port']}/{data_source['sid']}")
            df = pd.read_sql(f"SELECT * FROM {data_source['table']}", engine)
            collection_name = data_source['table']
        else:
            if data_source['file'] is not None:
                df = pd.read_csv(data_source['file'])
                collection_name = os.path.splitext(os.path.basename(data_source['file'].name))[0]
            else:
                st.error('Please upload a data file.')
                return
        # Chunking (placeholder)
        texts = df.iloc[:,0].astype(str).tolist()
        # TODO: Add chunking logic based on embed_config['chunking_strategy'] and chunk_size
        # Embedding
        model = SentenceTransformer(embed_config['model'])
        start_embed = time.time()
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=embed_config['batch_size'])
        end_embed = time.time()
        embedding_time = end_embed - start_embed
        # Insert and search logic for Milvus and Qdrant
        if db_type == 'Milvus':
            collection = ensure_milvus_collection(collection_name, dim=embed_config['dimension'])
            ids = list(range(len(embeddings)))
            data_to_insert = [ids, embeddings.tolist()]
            collection.insert(data_to_insert)
            # Retrieval
            start_response = time.time()
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = collection.search([embeddings[0]], "embedding", search_params, limit=5)
            end_response = time.time()
            response_time = end_response - start_response
            retrieved_ids = [hit.id for hit in results[0]] if results and len(results) > 0 else []
        elif db_type == 'Qdrant':
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import VectorParams, Distance
            client = QdrantClient(host=host, port=int(port))
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embed_config['dimension'], distance=Distance.COSINE)
            )
            payload = [
                {
                    "id": i,
                    "vector": emb.tolist(),
                    "payload": {"text": text}
                }
                for i, (emb, text) in enumerate(zip(embeddings, texts))
            ]
            client.upsert(collection_name=collection_name, points=payload)
            start_response = time.time()
            search_res = client.search(collection_name=collection_name, query_vector=embeddings[0].tolist(), limit=5, with_payload=True)
            end_response = time.time()
            response_time = end_response - start_response
            retrieved_ids = [hit.id for hit in search_res]
        else:
            st.error(f"Vector DB type '{db_type}' not yet implemented.")
            return
        # Evaluation metrics
        relevant_ids = set(range(5))  # Assume first 5 are relevant
        true_positives = len(set(retrieved_ids) & relevant_ids)
        accuracy = true_positives / 5 if 5 else 0
        recall = true_positives / len(relevant_ids) if relevant_ids else 0
        precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        st.success('Embedding and evaluation complete!')
        st.metric('Accuracy', f'{accuracy:.3f}')
        st.metric('Recall', f'{recall:.3f}')
        st.metric('Precision', f'{precision:.3f}')
        st.metric('F1 Score', f'{f1:.3f}')
        st.metric('Embedding Time (s)', f'{embedding_time:.2f}')
        st.metric('Response Time (s)', f'{response_time:.2f}')

if __name__ == '__main__':
    main()

# Docker commands to pull and run Milvus
os.system("docker pull milvusdb/milvus:v2.3.9")
os.system("docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:v2.3.9")
