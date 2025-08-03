# Configuration Parameter Reference

This document provides a comprehensive reference for all configuration parameters in the embedding evaluation system.

## Configuration Structure

The system uses a hierarchical YAML configuration structure:

```yaml
vector_db:
  type: qdrant  # or milvus
  vector_index:
    type: hnsw
    params:
      m: 16
      ef_construct: 256
  
retrieval:
  score_threshold: 0.6
  rerank_results: true
  diversity_weight: 0.3
  metric_type: COSINE

embed_config:
  normalize: true
  batch_size: 32
```

## Vector Database Configuration

### General Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| type | string | "qdrant" | Vector database type (qdrant/milvus) |
| host | string | "localhost" | Database host address |
| port | integer | 6333/19530 | Database port |

### Vector Index Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| type | string | "hnsw" | Index type (hnsw/plain) |
| m | integer | 16 | HNSW connections per element |
| ef_construct | integer | 256 | HNSW build-time quality |
| ef_search | integer | 128 | HNSW query-time quality |

### Collection Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| indexing_threshold | integer | 20000 | Optimization threshold |
| memmap_threshold | integer | 20000 | Memory mapping threshold |

## Retrieval Configuration

### Search Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| score_threshold | float | 0.6 | Minimum similarity score |
| rerank_results | boolean | true | Enable result reranking |
| diversity_weight | float | 0.3 | Diversity penalty weight |
| metric_type | string | "COSINE" | Distance metric type |

### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| top_k | integer | 10 | Number of results to return |
| batch_size | integer | 32 | Query batch size |
| with_payload | boolean | true | Include payload in results |

## Embedding Configuration

### Model Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| normalize | boolean | true | Normalize embeddings |
| batch_size | integer | 32 | Embedding batch size |
| model_type | string | "sentence-transformer" | Embedding model type |

### Processing Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_length | integer | 512 | Maximum sequence length |
| truncation | boolean | true | Enable text truncation |
| padding | boolean | true | Enable padding |

## Performance Configuration

### Batch Processing
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| insertion_batch_size | integer | 100 | Database insertion batch size |
| query_batch_size | integer | 32 | Query processing batch size |

### Resource Management
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_memory | string | "4GB" | Maximum memory usage |
| num_threads | integer | 4 | Number of processing threads |

## Example Configurations

### High Performance Configuration
```yaml
vector_db:
  type: qdrant
  vector_index:
    type: hnsw
    params:
      m: 32
      ef_construct: 512
      ef_search: 256

retrieval:
  score_threshold: 0.5
  rerank_results: true
  diversity_weight: 0.2
  metric_type: COSINE

embed_config:
  normalize: true
  batch_size: 64
```

### Memory-Optimized Configuration
```yaml
vector_db:
  type: qdrant
  vector_index:
    type: hnsw
    params:
      m: 8
      ef_construct: 128
      ef_search: 64

retrieval:
  score_threshold: 0.7
  rerank_results: false
  diversity_weight: 0.0
  metric_type: COSINE

embed_config:
  normalize: true
  batch_size: 16
```

## Configuration Tips

1. **Vector Index**
   - Higher `m` and `ef_construct` for better accuracy
   - Lower values for faster indexing and queries
   - Balance based on dataset size and requirements

2. **Retrieval**
   - Adjust `score_threshold` based on precision needs
   - Enable `rerank_results` for better diversity
   - Tune `diversity_weight` based on use case

3. **Embedding**
   - Enable `normalize` for consistent results
   - Adjust `batch_size` based on available memory
   - Choose appropriate `model_type` for use case

4. **Performance**
   - Configure batch sizes based on system resources
   - Monitor and adjust thread count
   - Set appropriate memory limits
