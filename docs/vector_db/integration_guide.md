# Vector Database Integration Guide

This document provides detailed information about the vector database integrations available in the embedding evaluation system.

## Supported Databases

### 1. Milvus
- **Schema Configuration**: Dynamic schema creation based on payload structure
- **Data Types**: Support for INT64, FLOAT_VECTOR, VARCHAR, and FLOAT fields
- **Indexing**: IVF_FLAT index with L2 distance metric
- **Batch Processing**: Bulk insertion with automatic schema adaptation

### 2. Qdrant
- **Collection Management**: Automatic collection creation and status monitoring
- **Vector Configuration**: Configurable vector size and distance metrics (COSINE)
- **Batch Processing**: Configurable batch size with progress tracking
- **Payload Handling**: Automatic type conversion for compatibility

## Integration Details

### Milvus Integration

```python
def insert_embeddings_milvus(embeddings, ids, texts, payloads, collection_name, dim, host='localhost', port=19530):
    # Dynamic schema creation based on payload structure
    # Automatic index creation (IVF_FLAT)
    # Batch insertion with progress tracking
```

Key Features:
- Automatic schema inference from payload structure
- Dynamic field type mapping
- Built-in index optimization
- Connection management

### Qdrant Integration

```python
def insert_embeddings_qdrant(embeddings, texts, payloads, collection_name, vector_size, host='localhost', port=6333):
    # Automatic collection creation
    # HNSW index configuration
    # Batched insertion with monitoring
```

Key Features:
- HNSW index optimization
- Payload sanitization and type conversion
- Progress monitoring and metrics collection
- Robust error handling

## Indexing Configuration

### HNSW Index Parameters
- `m`: Number of connections per element (default: 16)
- `ef_construct`: Build-time quality parameter (default: 256)
- `indexing_threshold`: Optimization threshold (20000)
- `memmap_threshold`: Memory mapping threshold (20000)

## Usage Examples

### Milvus Example
```python
from vector_databases.insertion import insert_embeddings_milvus

# Insert embeddings
collection = insert_embeddings_milvus(
    embeddings=embeddings,
    ids=ids,
    texts=texts,
    payloads=payloads,
    collection_name="my_collection",
    dim=768  # embedding dimension
)
```

### Qdrant Example
```python
from vector_databases.insertion import insert_embeddings_qdrant

# Insert embeddings
client, metrics = insert_embeddings_qdrant(
    embeddings=embeddings,
    texts=texts,
    payloads=payloads,
    collection_name="my_collection",
    vector_size=768
)
```

## Performance Considerations

1. **Batch Size Optimization**
   - Default batch size: 100
   - Adjust based on memory constraints and insertion speed requirements

2. **Index Configuration**
   - HNSW parameters affect build time vs search performance
   - Tune based on dataset size and query requirements

3. **Memory Management**
   - Use batch processing for large datasets
   - Monitor memory usage during insertion

## Error Handling

Both integrations include robust error handling:
- Connection error management
- Type conversion error handling
- Batch insertion failure recovery
- Collection status monitoring

## Monitoring and Metrics

Available metrics include:
- Insertion time
- Index build time
- Insert rate (vectors/second)
- Collection status
- Batch processing statistics
