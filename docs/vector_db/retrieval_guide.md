# Retrieval System Documentation

This document details the retrieval capabilities and functionality of the embedding evaluation system.

## Overview

The system provides advanced retrieval mechanisms with:
- Vector similarity search
- Semantic search with query embedding
- Result reranking with diversity consideration
- Configurable search parameters
- Performance metrics tracking

## Retrieval Methods

### 1. Basic Vector Similarity Search
```python
def retrieve_milvus(col, embeddings, top_k, metric_type, nprobe):
    # Simple vector similarity search
    # Returns top-k nearest neighbors
```

### 2. Enhanced Semantic Search
```python
def retrieve_qdrant_with_results(client, collection, semantic_query, embed_model, top_k,
                               config=None, payload_filter_cfg=None, query_embedding=None):
    # Advanced semantic search with reranking and diversity
    # Returns results with detailed metrics
```

## Key Features

### 1. Query Processing
- Support for both vector and semantic queries
- Query embedding generation with normalization
- Configurable similarity metrics (COSINE, L2)

### 2. Result Ranking
- Score-based initial ranking
- Diversity-aware reranking
- Configurable score thresholds
- Content-based deduplication

### 3. Performance Optimization
- HNSW search parameter tuning
- Configurable search efficiency (ef_search)
- Query timing and throughput metrics

## Configuration Options

### Search Parameters
- `score_threshold`: Minimum similarity score (default: 0.6)
- `rerank_results`: Enable/disable reranking (default: true)
- `diversity_weight`: Weight for diversity penalty (default: 0.3)
- `top_k`: Number of results to return

### Index Parameters
- `ef_search`: HNSW search complexity parameter
- `metric_type`: Distance metric for similarity calculation
- `exact`: Enable/disable exact search

## Result Diversity

The system implements diversity-aware search through:
1. Content fingerprinting
2. Diversity penalties
3. Reranking with combined scores
4. Unique content tracking

## Performance Metrics

The system tracks:
1. Query execution time
2. Query throughput
3. Result similarity scores
4. Result diversity scores

## Example Usage

### Basic Vector Search
```python
from retrieval.retriever import retrieve_milvus

results = retrieve_milvus(
    col=collection,
    embeddings=query_embeddings,
    top_k=10,
    metric_type="L2",
    nprobe=10
)
```

### Advanced Semantic Search
```python
from retrieval.retriever import retrieve_qdrant_with_results

results, metrics = retrieve_qdrant_with_results(
    client=qdrant_client,
    collection="my_collection",
    semantic_query="example query",
    embed_model=model,
    top_k=10,
    config=search_config
)
```

## Error Handling

The retrieval system includes:
- Query validation
- Result verification
- Error logging
- Performance monitoring

## Best Practices

1. **Query Optimization**
   - Use appropriate score thresholds
   - Balance diversity vs relevance
   - Monitor query performance

2. **Result Quality**
   - Enable reranking for better diversity
   - Adjust diversity weight based on use case
   - Use payload filters when needed

3. **Performance Tuning**
   - Configure ef_search based on dataset size
   - Monitor and adjust score thresholds
   - Use pre-computed query embeddings when possible
