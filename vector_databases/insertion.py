# Milvus and Qdrant support

def insert_embeddings_milvus(embeddings, ids, collection, dimension, host, port):
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
    connections.connect(host=host, port=port)
    if collection in [c.name for c in Collection.list()]:
        col = Collection(collection)
        col.drop()
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=dimension)
    ]
    schema = CollectionSchema(fields, description="Embedding collection")
    col = Collection(collection, schema)
    data_to_insert = [ids, [emb.tolist() for emb in embeddings]]
    col.insert(data_to_insert)
    return col

def insert_embeddings_qdrant(embeddings, ids, texts, collection, dimension, host, port):
    import logging
    import time
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams, Distance
    from tqdm import trange
    import inspect
    client = QdrantClient(host=host, port=port)
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )
    # Pull batch size and retry config from caller's config if available
    batch_size = 500
    upsert_retries = 3
    upsert_retry_delay = 2
    frame = inspect.currentframe().f_back
    if 'vdb_cfg' in frame.f_locals:
        vdb_cfg = frame.f_locals['vdb_cfg']
        batch_size = int(vdb_cfg.get('batch_size', 500))
        upsert_retries = int(vdb_cfg.get('upsert_retries', 3))
        upsert_retry_delay = int(vdb_cfg.get('upsert_retry_delay', 2))
    total = len(embeddings)
    for start in trange(0, total, batch_size, desc=f"[Qdrant] Upserting in batches of {batch_size}"):
        end = min(start + batch_size, total)
        payload = [
            {
                "id": ids[i],
                "vector": emb.tolist(),
                "payload": {"text": texts[i]}
            }
            for i, emb in enumerate(embeddings[start:end], start=start)
        ]
        success = False
        for attempt in range(1, upsert_retries + 1):
            try:
                client.upsert(collection_name=collection, points=payload)
                logging.info(f"[Qdrant] Upserted batch {start}-{end} ({end-start} points) on attempt {attempt}")
                success = True
                break
            except Exception as e:
                logging.error(f"[Qdrant][Error] Upsert batch {start}-{end} attempt {attempt} failed: {e}")
                if attempt < upsert_retries:
                    logging.info(f"[Qdrant] Retrying batch {start}-{end} in {upsert_retry_delay} seconds...")
                    time.sleep(upsert_retry_delay)
        if not success:
            logging.error(f"[Qdrant][Error] Upsert batch {start}-{end} failed after {upsert_retries} attempts. Aborting.")
            raise Exception(f"Qdrant upsert failed for batch {start}-{end} after {upsert_retries} attempts.")
    return client
