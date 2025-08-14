import logging
import time
from typing import List, Dict, Tuple, Any
from tqdm import trange
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, 
    Distance, 
    VectorParams,
    CollectionStatus,
    SearchRequest,
    Filter,
    FieldCondition,
    Range,
    MatchValue,
    OptimizersConfigDiff,
    HnswConfigDiff
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
    timeout: int = 300,
    check_interval: int = 1
) -> None:
    """
    Ensure the collection exists, create if it doesn't
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        vector_size: Size of vectors
        timeout: Operation timeout in seconds
        check_interval: Interval between status checks in seconds
    """
    try:
        # Sanitize collection name - remove file path elements
        collection_name = collection_name.replace('/', '_').replace('\\', '_')
        
        # Set client timeout
        client.timeout = timeout
        
        # Check if collection exists with timeout handling
        start_time = time.time()
        while True:
            try:
                collections = client.get_collections().collections
                collection_exists = any(col.name == collection_name for col in collections)
                break
            except Exception as e:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Collection check timed out after {timeout} seconds")
                time.sleep(check_interval)
                logging.warning(f"Collection check failed, retrying... Error: {str(e)}")

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
            
            # Wait for collection to be ready with timeout
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Collection creation timed out after {timeout} seconds")
                    
                try:
                    collection_info = client.get_collection(collection_name)
                    if collection_info.status == CollectionStatus.GREEN:
                        break
                except Exception as e:
                    logging.warning(f"Collection status check failed: {str(e)}")
                    
                time.sleep(check_interval)
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
            
            # Debug output for each point
            logging.debug(f"\nEmbedding ID: {idx}")
            logging.debug(f"Vector dimension: {len(vector)}")
            logging.debug(f"Payload: {sanitized_payload}")
            
            point = PointStruct(
                id=idx,
                vector=vector,
                payload=sanitized_payload
            )
            points.append(point)
            
            # Print embedding details once at the start
            if idx == 0:
                logging.info("\n=== Embedding Structure Example ===")
                logging.info(f"Vector dimension: {len(vector)}")
                logging.info("Payload structure:")
                for key in sanitized_payload.keys():
                    logging.info(f"- {key}")
                logging.info("=" * 50)
            
            # Print concise embedding info
            logging.info(f"[{idx}] ID: {idx}, "
                        f"Text: {sanitized_payload.get('text', '')[:50]}{'...' if len(sanitized_payload.get('text', '')) > 50 else ''}, "
                        f"Metadata: {', '.join([f'{k}={v}' for k, v in sanitized_payload.items() if k != 'text'])}")
        
        # Insert in batches with progress bar
        total_batches = (len(points) + batch_size - 1) // batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                # Print batch details
                logging.info(f"\n--- Batch {i//batch_size + 1}/{total_batches} ---")
                logging.info(f"Size: {len(batch)} points | ID range: {batch[0].id} to {batch[-1].id}")
                
                client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True  # Wait for operation to complete
                )
                
                # Log batch metrics
                logging.info(f"Status: Successfully inserted {len(batch)} points")
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

    try:
        # Configure vector index
        index_type = vector_index_cfg.get('type', 'hnsw').lower()
        if index_type == 'hnsw':
            params = vector_index_cfg.get('params', {})
            
            # Update HNSW config
            optimizer_config = OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=20000,
            )
            
            hnsw_config = HnswConfigDiff(
                m=params.get('m', 16),
                ef_construct=params.get('ef_construct', 256),
            )
            
            # Update collection with new configuration
            client.update_collection(
                collection_name=collection_name,
                optimizers_config=optimizer_config,
                hnsw_config=hnsw_config
            )
            
            # Note: Qdrant's update_collection does not accept 'params'.
            # If you want to set search parameters like 'hnsw_ef', you must do it at query time, not as a collection config.
            
            # Configure quantization if enabled (Qdrant expects a specific structure)
            quant_cfg = vector_index_cfg.get('quantization', {})
            if quant_cfg.get('enabled', False):
                from qdrant_client.models import ScalarQuantization
                scalar_quant = ScalarQuantization(
                    scalar={
                        "type": "int8",
                        "always_ram": quant_cfg.get('always_ram', True)
                    }
                )
                client.update_collection(
                    collection_name=collection_name,
                    quantization_config=scalar_quant
                )
        
        # Configure payload indexes
        supported_types = {'keyword', 'integer', 'float', 'geo', 'text', 'bool', 'datetime', 'uuid'}
        for field_cfg in payload_index_cfg:
            field_name = field_cfg['field']
            field_type = field_cfg['type']
            if field_type not in supported_types:
                logging.warning(f"[Qdrant] Skipping unsupported payload index type: {field_type} for field {field_name}")
                continue
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type
            )
                    
        logging.info(f"Successfully configured indexes for collection {collection_name}")
        
    except Exception as e:
        logging.error(f"Error configuring indexes: {str(e)}")
        raise


def insert_vectors(collection, embeddings, texts, payloads, batch_size=100) -> Dict[str, float]:
    """
    Insert vectors into the collection with their corresponding payloads.
    
    Args:
        collection: The vector database collection
        embeddings: List of embedding vectors
        texts: List of original texts
        payloads: List of payload dictionaries containing original column values
        batch_size: Size of batches for insertion
        
    Returns:
        Dictionary containing insertion metrics
    """
    start_time = time.time()
    total_vectors = len(embeddings)
    metrics = {}
    
    try:
        # Log start of insertion
        logging.info(f"Starting insertion of {total_vectors} vectors in batches of {batch_size}")
        
        # Calculate total batches for progress tracking
        total_batches = (total_vectors + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, total_vectors, batch_size), 1):
            batch_start_time = time.time()
            batch_end = min(i + batch_size, total_vectors)
            
            # Create batch with proper ID generation and payload handling
            entities = []
            for j in range(i, batch_end):
                # Convert numpy array to list if necessary
                vector = embeddings[j].tolist() if hasattr(embeddings[j], 'tolist') else embeddings[j]
                
                # Ensure payload values are of supported types
                sanitized_payload = {"text": str(texts[j])}
                if payloads and j < len(payloads):
                    for k, v in payloads[j].items():
                        if isinstance(v, (int, float, str, bool)):
                            sanitized_payload[k] = v
                        else:
                            sanitized_payload[k] = str(v)
                
                entity = {
                    "id": str(j),
                    "vector": vector,
                    "payload": sanitized_payload
                }
                entities.append(entity)
                
                # Debug output for every 100th vector
                if j % 100 == 0:
                    logging.debug(f"\n=== Vector {j} Details ===")
                    logging.debug(f"ID: {j}")
                    logging.debug(f"Vector dimension: {len(vector)}")
                    logging.debug(f"Payload keys: {list(sanitized_payload.keys())}")
            
            try:
                # Insert batch with progress logging
                logging.info(f"Inserting batch {batch_num}/{total_batches} "
                           f"(vectors {i} to {batch_end-1})")
                collection.insert(entities)
                
                batch_time = time.time() - batch_start_time
                logging.info(f"Batch {batch_num} inserted in {batch_time:.2f}s "
                           f"({len(entities)/batch_time:.1f} vectors/s)")
                
            except Exception as e:
                logging.error(f"Error inserting batch {batch_num}: {str(e)}")
                raise
        
        # Calculate and return metrics
        insertion_time = time.time() - start_time
        metrics = {
            'insertion_time': insertion_time,
            'total_vectors': total_vectors,
            'insert_rate': total_vectors / insertion_time if insertion_time > 0 else 0,
            'batch_size': batch_size
        }
        
        logging.info(f"Insertion completed: {metrics['total_vectors']} vectors "
                    f"in {metrics['insertion_time']:.2f}s "
                    f"({metrics['insert_rate']:.1f} vectors/s)")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error in vector insertion: {str(e)}")
        raise
