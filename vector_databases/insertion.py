import logging
import time
from typing import List, Dict, Tuple, Any
from tqdm import trange
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, 
    Distance, 
    VectorParams,
    CollectionStatus
)

# Milvus and Qdrant support

def insert_embeddings_milvus(embeddings, ids, texts, payloads, collection_name, dim, host='localhost', port=19530):
    from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema, DataType
    
    connections.connect(host=host, port=port)
    
    # Define collection schema with payload fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    
    # Add payload fields dynamically based on the first payload
    if payloads and len(payloads) > 0:
        sample_payload = payloads[0]
        for key, value in sample_payload.items():
            if isinstance(value, (int, float)):
                fields.append(FieldSchema(name=key, dtype=DataType.FLOAT))
            else:
                fields.append(FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=65535))
    
    schema = CollectionSchema(fields)
    
    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    
    # Prepare data for insertion
    entities = [
        {
            "id": ids,
            "embedding": embeddings,
            "text": texts,
            **{k: [p.get(k) for p in payloads] for k in payloads[0].keys()}
        }
    ]
    
    # Insert data
    collection.insert(entities)
    collection.flush()
    
    # Build index if needed
    collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}})
    
    return collection

def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """
    Ensure the collection exists, create if it doesn't
    """
    try:
        # Sanitize collection name - remove file path elements
        collection_name = collection_name.replace('/', '_').replace('\\', '_')
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)

        if not collection_exists:
            # Create collection if it doesn't exist
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logging.info(f"Created new collection: {collection_name}")
            
            # Wait for collection to be ready
            while True:
                collection_info = client.get_collection(collection_name)
                if collection_info.status == CollectionStatus.GREEN:
                    break
                time.sleep(1)
                logging.info("Waiting for collection to be ready...")
            
            logging.info(f"Collection {collection_name} is ready")
    except Exception as e:
        logging.error(f"Error ensuring collection exists: {str(e)}")
        raise

def insert_embeddings_qdrant(
    embeddings: List[Any],
    texts: List[str],
    payloads: List[Dict],
    collection_name: str,
    vector_size: int,
    host: str = 'localhost',
    port: int = 6333,
    batch_size: int = 100
) -> Tuple[QdrantClient, float]:
    """
    Insert embeddings into Qdrant vector database
    
    Args:
        embeddings: List of embedding vectors
        texts: List of original texts
        payloads: List of payload dictionaries
        collection_name: Name of the collection
        vector_size: Size of the embedding vectors
        host: Qdrant host
        port: Qdrant port
        batch_size: Batch size for insertion
    
    Returns:
        Tuple of (QdrantClient, insertion_time)
    """
    start_time = time.time()
    metrics = {}
    
    try:
        client = QdrantClient(host=host, port=port)
        
        # Record collection creation/index build time
        index_start_time = time.time()
        ensure_collection_exists(client, collection_name, vector_size)
        metrics['index_build_time'] = time.time() - index_start_time
        
        # Prepare points with payloads
        points = []
        for idx, (emb, text, payload) in enumerate(zip(embeddings, texts, payloads)):
            # Convert numpy array to list if necessary
            vector = emb.tolist() if hasattr(emb, 'tolist') else emb
            
            # Ensure payload values are of supported types
            sanitized_payload = {"text": str(text)}
            for k, v in payload.items():
                # Convert values to supported types
                if isinstance(v, (int, float, str, bool)):
                    sanitized_payload[k] = v
                else:
                    sanitized_payload[k] = str(v)
            
            point = PointStruct(
                id=idx,
                vector=vector,
                payload=sanitized_payload
            )
            points.append(point)
        
        # Insert in batches with progress bar
        total_batches = (len(points) + batch_size - 1) // batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True  # Wait for operation to complete
                )
                logging.info(f"Inserted batch {i//batch_size + 1}/{total_batches}")
            except Exception as e:
                logging.error(f"Error inserting batch {i//batch_size + 1}: {str(e)}")
                raise
        
        insertion_time = time.time() - start_time
        total_vectors = len(embeddings)
        metrics.update({
            'insertion_time': insertion_time,
            'insert_rate': total_vectors / insertion_time if insertion_time > 0 else 0
        })
        return client, metrics
        
    except Exception as e:
        logging.error(f"Error inserting into Qdrant: {str(e)}")
        raise

def setup_qdrant_indexing(
    client: QdrantClient,
    collection_name: str,
    vector_index_cfg: Dict,
    payload_index_cfg: List[Dict]
) -> None:
    """
    Configure Qdrant indexing settings
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        vector_index_cfg: Vector index configuration
        payload_index_cfg: Payload index configuration
    """
    try:
        # Update collection settings
        if vector_index_cfg:
            index_type = vector_index_cfg.get('type', 'hnsw')
            index_params = vector_index_cfg.get('params', {})
            
            hnsw_config = None
            if index_type == 'hnsw':
                hnsw_config = {
                    k: v for k, v in index_params.items() 
                    if k in ['m', 'ef_construct']
                }
            
            client.update_collection(
                collection_name=collection_name,
                optimizer_config={"default_segment_number": 2},
                hnsw_config=hnsw_config
            )
            logging.info(f"Updated collection indexing settings: {collection_name}")

        # Create payload indices
        for field_cfg in payload_index_cfg:
            field = field_cfg.get('field')
            idx_type = field_cfg.get('type')
            if field and idx_type:
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=idx_type
                    )
                    logging.info(f"Created payload index for field: {field}")
                except Exception as e:
                    logging.warning(f"Failed to create payload index for {field}: {str(e)}")

    except Exception as e:
        logging.error(f"Error setting up Qdrant indexing: {str(e)}")
        raise

def insert_vectors(collection, embeddings, texts, payloads, batch_size=100):
    """
    Insert vectors into the collection with their corresponding payloads.
    
    Args:
        collection: The vector database collection
        embeddings: List of embedding vectors
        texts: List of original texts
        payloads: List of payload dictionaries containing original column values
        batch_size: Size of batches for insertion
    """
    total_vectors = len(embeddings)
    
    for i in range(0, total_vectors, batch_size):
        batch_end = min(i + batch_size, total_vectors)
        
        entities = [
            {
                "id": str(j),
                "vector": embeddings[j],
                "payload": {
                    "text": texts[j],
                    **payloads[j]  # Unpack all original columns into payload
                }
            }
            for j in range(i, batch_end)
        ]
        
        collection.insert(entities)
