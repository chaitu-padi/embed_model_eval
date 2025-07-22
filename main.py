import argparse
import os
import yaml
import time
from data_sources.loader import load_data
from embedding_models.embedder import embed_texts
from vector_databases.insertion import insert_embeddings_milvus, insert_embeddings_qdrant
from retrieval.retriever import retrieve_milvus, retrieve_qdrant
from evaluation.metrics import evaluate
from reporting.report import print_report
from retrieval.retriever import retrieve_qdrant_with_results

def main():
    parser = argparse.ArgumentParser(description="Embedding Model Evaluation CLI")
    parser.add_argument('--config', type=str, default='../config/config.yaml', help='Path to configuration YAML file')
    args = parser.parse_args()

    import logging
    from tqdm import tqdm
    import pprint
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info("\n[Process] Starting Embedding Model Evaluation Pipeline...")
    from common_utils.config_utils import get_config_path, load_yaml_config
    config_path = args.config if os.path.isabs(args.config) else get_config_path(os.path.basename(args.config))
    logging.info(f"[Config] Loading configuration from: {config_path}")
    config = load_yaml_config(config_path)
    logging.info("[Config] Loaded configuration parameters:")
    logging.info("\n" + pprint.pformat(config, indent=2))

    # 1. Data Source
    logging.info("\n[Step 1] Data Source Loading...")
    ds = config['data_source']
    # Update data source path for new data folder
    if config['data_source']['type'] == 'csv':
        # If the file path in YAML is not absolute, resolve it relative to the data/output folder
        file_path = config['data_source']['file']
        if not os.path.isabs(file_path):
            # Assume files are inside data/output/
            config['data_source']['file'] = os.path.join(
            os.path.dirname(__file__), 'data', 'output', os.path.basename(file_path)
            )
    logging.info(f"[Data] Data source type: {ds['type']}")
    logging.info(f"[Data] Data source file/table: {ds.get('file', ds.get('table', ''))}")
    df, collection_name, prepared_texts = load_data(config)

    # 2. Chunking
    chunk_cfg = config.get('chunking', {})
    def chunk_texts(texts, strategy, chunk_size, overlap, delimiter):
        logging.info(f"[Chunking] Strategy: {strategy}, Chunk Size: {chunk_size}, Overlap: {overlap}, Delimiter: {delimiter}")
        if strategy == 'none':
            logging.info(f"[Chunking] No chunking applied. Returning original texts.")
            return texts
        elif strategy == 'sentence':
            import nltk
            # Try to find both 'punkt' and 'punkt_tab', download if missing
            for resource in ['punkt', 'punkt_tab']:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    logging.info(f"[NLTK] '{resource}' not found. Downloading...")
                    nltk.download(resource)
            from nltk.tokenize import sent_tokenize
            chunked = []
            for text in texts:
                try:
                    sentences = sent_tokenize(str(text))
                    chunked.extend(sentences)
                except Exception as e:
                    logging.error(f"[Chunking][Error] Failed to tokenize text: {text[:50]}... Error: {e}")
            logging.info(f"[Chunking] Total sentences after chunking: {len(chunked)}")
            return chunked
        elif strategy == 'sliding_window':
            chunks = []
            for text in texts:
                for i in range(0, len(text), chunk_size - overlap):
                    chunks.append(text[i:i+chunk_size])
            logging.info(f"[Chunking] Total chunks after sliding window: {len(chunks)}")
            return chunks
        elif strategy == 'fixed_length':
            chunks = [text[i:i+chunk_size] for text in texts for i in range(0, len(text), chunk_size)]
            logging.info(f"[Chunking] Total chunks after fixed length: {len(chunks)}")
            return chunks
        else:
            logging.error(f"[Chunking][Error] Unsupported chunking strategy: {strategy}")
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

    logging.info(f"[Data] Loaded {len(df)} rows from data source.")
    if prepared_texts is not None:
        texts = prepared_texts
        logging.info(f"[Data] Used best-practice concatenation for {len(texts)} rows and columns: {config['data_source'].get('embed_columns', 'all')}")
    else:
        texts = df.iloc[:,0].astype(str).tolist()
        logging.info(f"[Data] Extracted {len(texts)} texts for embedding (default first column).")
    logging.info("\n[Step 2] Chunking Texts...")
    texts = chunk_texts(
        texts,
        chunk_cfg.get('strategy', 'none'),
        int(chunk_cfg.get('chunk_size', 128)),
        int(chunk_cfg.get('overlap', 0)),
        chunk_cfg.get('delimiter', '\n')
    )

    # 3. Embedding
    logging.info("\n[Step 3] Embedding Generation...")
    emb_cfg = config['embed_config']
    logging.info(f"[Embedding] Model: {emb_cfg.get('model', 'all-MiniLM-L6-v2')}")
    logging.info(f"[Embedding] Batch Size: {emb_cfg.get('batch_size', 64)} | Parallelism: {emb_cfg.get('parallelism', 1)} | Dimension: {emb_cfg.get('dimension', 768)} | Normalize: {emb_cfg.get('normalize', True)}")
    start_embed = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(emb_cfg.get('model', 'all-MiniLM-L6-v2'))
    batch_size = int(emb_cfg.get('batch_size', 64))
    normalize = emb_cfg.get('normalize', True)
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    embeddings = []
    for i, batch in enumerate(tqdm(batches, desc="[Embedding] Generating embeddings", unit="batch")):
        emb = model.encode(batch, show_progress_bar=False, batch_size=batch_size, normalize_embeddings=normalize)
        embeddings.extend(emb)
        logging.info(f"[Embedding] Progress: {min((i+1)*batch_size, len(texts))}/{len(texts)} texts embedded.")
    end_embed = time.time()
    embedding_time = end_embed - start_embed
    logging.info(f"[Embedding] Completed. Time taken: {embedding_time:.2f} seconds. Total embeddings: {len(embeddings)}")

    # 4. Vector DB Ingestion
    logging.info("\n[Step 4] Vector Database Ingestion...")
    start_embed = time.time()
    vdb_cfg = config['vector_db']
    db_type = vdb_cfg.get('type', 'Milvus').lower()
    host = vdb_cfg.get('host', 'localhost')
    port = int(vdb_cfg.get('port', 19530 if db_type == 'milvus' else 6333))
    collection = vdb_cfg.get('collection', collection_name)
    ids = list(range(len(embeddings)))
    logging.info(f"[DB] Type: {db_type} | Host: {host} | Port: {port} | Collection: {collection}")
    logging.info(f"[DB] Inserting {len(embeddings)} embeddings...")
    from tqdm import trange
    if db_type == 'milvus':
        # Simulate progress bar for insertion
        col = None
        for i in trange(1, desc="[DB] Inserting into Milvus", total=1):
            col = insert_embeddings_milvus(embeddings, ids, collection, emb_cfg.get('dimension', 768), host, port)
        logging.info(f"[DB] Inserted {len(embeddings)} embeddings into Milvus collection '{collection}'")
    elif db_type == 'qdrant':
        # Simulate progress bar for insertion
        client = None
        for i in trange(1, desc="[DB] Inserting into Qdrant", total=1):
            client = insert_embeddings_qdrant(embeddings, ids, texts, collection, emb_cfg.get('dimension', 768), host, port)
        logging.info(f"[DB] Inserted {len(embeddings)} embeddings into Qdrant collection '{collection}'")
    else:
        logging.error(f"[DB][Error] Unsupported vector DB type: {db_type}")
        raise ValueError(f"Unsupported vector DB type: {db_type}")
    end_embed = time.time()
    embedding_insertion_time = end_embed - start_embed
    logging.info(f"[Embedding Insertion] Completed. Time taken: {embedding_insertion_time:.2f} seconds. Total embeddings: {len(embeddings)}")

    # --- Qdrant Indexing Setup ---
    if db_type == 'qdrant':
        vector_index_cfg = vdb_cfg.get('vector_index', {})
        payload_index_cfg = vdb_cfg.get('payload_index', [])
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port)
        # Vector index setup
        if vector_index_cfg:
            index_type = vector_index_cfg.get('type', 'hnsw')
            index_params = vector_index_cfg.get('params', {})
            optimizer_config = {"default_segment_number": 1}
            hnsw_config = None
            if index_type == 'hnsw':
                # Only pass HNSW params to hnsw_config
                hnsw_config = {k: v for k, v in index_params.items() if k in ['m', 'ef_construct']}
            # Remove unsupported fields from optimizer_config
            client.update_collection(
                collection_name=collection,
                optimizer_config=optimizer_config,
                hnsw_config=hnsw_config
            )
            logging.info(f"[Qdrant] Vector index configured: {index_type} with hnsw_config {hnsw_config}")
        # Payload index setup
        for field_cfg in payload_index_cfg:
            field = field_cfg.get('field')
            idx_type = field_cfg.get('type')
            if field and idx_type:
                client.create_payload_index(collection_name=collection, field_name=field, field_type=idx_type)
                logging.info(f"[Qdrant] Payload index created for field '{field}' of type '{idx_type}'")

    # 5. Retrieval & Evaluation
    ret_cfg = config.get('retrieval', {})
    eval_cfg = config.get('evaluation', {})
    top_k = int(ret_cfg.get('top_k', 5))
    metric_type = ret_cfg.get('metric_type', 'L2')
    nprobe = int(ret_cfg.get('nprobe', 10))
    distance = ret_cfg.get('distance', 'COSINE')
    filter_expr = ret_cfg.get('filter', '')

    logging.info("\n[Step 5] Retrieval & Evaluation...")
    logging.info(f"[Retrieval] Starting retrieval for top_k={top_k}...")
    start_retrieve = time.time()
    if db_type == 'milvus':
        retrieved_ids = retrieve_milvus(col, embeddings, top_k, metric_type, nprobe)
        logging.info(f"[Retrieval] Retrieved IDs from Milvus: {retrieved_ids}")
    elif db_type == 'qdrant':
        semantic_query = ret_cfg.get('semantic_query', None)
        if semantic_query:   
            retrieved_ids=[]      
            print(f"\nSemantic Query: {ret_cfg['semantic_query']}")
            results = retrieve_qdrant_with_results(client, collection, ret_cfg['semantic_query'], model, top_k)
            print("\nRetrieved Results:")
            for rid, text in results:
                retrieved_ids.append(rid)
                print(f"ID: {rid}, Text: {text}")
            logging.info(f"[Retrieval] Retrieved IDs from Qdrant: {retrieved_ids}")
        else:
            retrieved_ids = retrieve_qdrant(client, collection, embeddings, top_k)
            logging.info(f"[Retrieval] Retrieved IDs from Qdrant: {retrieved_ids}")
    else:
        logging.error(f"[Retrieval][Error] Unsupported vector DB type: {db_type}")
        retrieved_ids = []
    end_retrieve = time.time()
    retrieval_time = end_retrieve - start_retrieve

    logging.info(f"[Evaluation] Evaluating retrieval results...")



    # 6. Report generation: Populate metrics with all required values for reporting
    metrics = evaluate(retrieved_ids, eval_cfg.get('relevant_ids', list(range(top_k))), top_k)
    metrics['embedding_time'] = embedding_time
    metrics['retrieval_time'] = retrieval_time
    metrics['total_embeddings'] = len(embeddings)
    metrics['batch_size'] = vdb_cfg.get('batch_size', 'N/A')
    metrics['upsert_retries'] = vdb_cfg.get('upsert_retries', 'N/A')
    metrics['insertion_time'] = embedding_insertion_time  
    metrics['dimension'] = emb_cfg.get('dimension', 'N/A')
    metrics['parallelism'] = emb_cfg.get('parallelism', 'N/A')
    metrics['top_k'] = top_k

    print_report(
        emb_cfg.get('model', 'all-MiniLM-L6-v2'),
        ds,
        db_type,
        host,
        port,
        collection,
        embedding_time,
        retrieval_time,
        top_k,
        metrics
    )
    logging.info("\n[Process] Embedding Model Evaluation Pipeline Complete.")

if __name__ == "__main__":
    main()
