# Embedding Model Evaluation Configuration Guide

This document provides a detailed guide to configuring your embedding model evaluation using the `config.yaml` file.

## Configuration Structure

The configuration file is divided into several main sections:
1. Data Source Configuration
2. Embedding Parameters
3. Chunking Parameters
4. Vector Database Configuration
5. Retrieval Configuration

## 1. Data Source Configuration

```yaml
data_source:
  type: pdf                           # Required: 'csv', 'oracle', 'json', 'txt', 'pdf'
  file: 'data/pdf_files/jesc*.pdf'   # Required: Path to data file(s)
  
  # PDF-specific settings (Required if type: pdf)
  pdf_config:
    extract_tables: true             # Optional: Extract tables from PDF (default: false)
    include_metadata: true           # Optional: Include PDF metadata (default: false)
    ocr_enabled: false              # Optional: Enable OCR for scanned docs (default: false)

  # CSV/Oracle-specific settings
  embed_columns: ["col1", "col2"]    # Optional: Columns to embed (default: all)
  
  # Oracle-specific settings (Required if type: oracle)
  host: localhost                    # Database host
  port: 1521                        # Database port
  sid: XE                           # Database SID
  user: username                    # Database username
  password: password                # Database password
  table: table_name                 # Source table name
```

### Dependencies and Requirements

- For `type: pdf`:
  - Must specify `file` with path to PDF file(s)
  - Can use wildcards (e.g., `*.pdf`) for multiple files
  - `pdf_config` section is required

- For `type: oracle`:
  - All Oracle connection parameters required
  - `embed_columns` is optional

## 2. Embedding Parameters

```yaml
embed_config:
  models:                          # Required: List of models to evaluate
    - "all-MiniLM-L6-v2"
    - "paraphrase-distilroberta-base-v1"
  
  default_model: "all-MiniLM-L6-v2"  # Required: Default model
  batch_size: 256                     # Required: Batch size for embedding
  normalize: true                     # Required: Whether to normalize embeddings
  
  optimization:                       # Optional: Performance settings
    use_gpu: true                    # Optional: Use GPU if available
    half_precision: true             # Optional: Use FP16 for GPU
    num_workers: 8                   # Optional: Number of workers
  
  dimension: 384                      # Required: Target embedding dimension
  use_pca: true                      # Optional: Use PCA reduction
  
  pca_config:                        # Required if use_pca: true
    random_state: 42                 # Optional: For reproducibility
    whiten: false                    # Optional: Scale components
  
  indexing_type: flat                # Required: 'flat', 'ivf', 'hnsw'
  embedding_type: semantic           # Required: 'semantic', 'dense', 'sparse', 'hybrid'
  search_type: semantic              # Required: 'semantic', 'dense', 'sparse', 'hybrid'
```

### Dependencies and Requirements

- When using multiple models:
  - All models must output same dimension or
  - `use_pca` must be true to normalize dimensions
  - `dimension` must be specified for PCA target

- GPU Optimization:
  - `use_gpu` requires CUDA-capable GPU
  - `half_precision` requires `use_gpu: true`

## 3. Chunking Parameters

```yaml
chunking:
  strategy: sentence                # Required: Chunking strategy
  chunk_size: 1000                 # Required: Target chunk size
  overlap: 100                     # Optional: Overlap between chunks
```

### Valid Strategy Values
- `none`: No chunking
- `sentence`: Split by sentences
- `sliding_window`: Overlapping chunks
- `fixed_length`: Fixed-size chunks
- `paragraph`: Split by paragraphs

### Dependencies
- If `strategy: sliding_window`:
  - `overlap` parameter required
  - `overlap` < `chunk_size`

## 4. Vector Database Configuration

```yaml
vector_db:
  type: qdrant                     # Required: 'qdrant' or 'milvus'
  host: localhost                  # Required: Database host
  port: 6333                      # Required: Database port
  collection: embeddings          # Optional: Collection name
  batch_size: 300                # Optional: Insertion batch size
  upsert_retries: 3             # Optional: Retry count on failure
  retry_delay: 2                # Optional: Seconds between retries
```

### Type-specific Requirements

For Qdrant:
- Default port: 6333
- Supports all index types

For Milvus:
- Default port: 19530
- Requires collection creation before use

## 5. Retrieval Configuration

```yaml
retrieval:
  top_k: 10                      # Required: Number of results
  distance_type: COSINE         # Required: COSINE, DOT, EUCLIDEAN
  filter_query: null           # Optional: Query filtering
  min_score: 0.5              # Optional: Minimum similarity
```

### Query Types
- `semantic`: Text-based semantic search
- `dense`: Vector similarity search
- `sparse`: Keyword-based search
- `hybrid`: Combined approach

### Dependencies
- For semantic search:
  - Embedding model must support text encoding
  - Query length within model limits

## Example Configurations

### Basic PDF Processing
```yaml
data_source:
  type: pdf
  file: "docs/*.pdf"
  pdf_config:
    extract_tables: true

embed_config:
  models: ["all-MiniLM-L6-v2"]
  batch_size: 256
  normalize: true
  dimension: 384

chunking:
  strategy: sentence
  chunk_size: 1000

vector_db:
  type: qdrant
  host: localhost
  port: 6333
```

### Multi-Model Evaluation
```yaml
data_source:
  type: csv
  file: "data.csv"
  embed_columns: ["text", "description"]

embed_config:
  models: 
    - "all-MiniLM-L6-v2"
    - "paraphrase-distilroberta-base-v1"
  use_pca: true
  dimension: 384
  normalize: true

vector_db:
  type: qdrant
  batch_size: 500
```

## Performance Considerations

1. Batch Size:
   - Larger batch_size → Better throughput
   - But also → Higher memory usage
   - Recommended: 256-512 for most cases

2. PCA Reduction:
   - Reduces embedding dimension
   - Improves storage/query speed
   - Some information loss
   - Use when different model dimensions

3. Chunking:
   - Smaller chunks → More precise retrieval
   - But also → More vectors to store/search
   - Balance based on content type

4. Vector DB:
   - Batch inserts for better performance
   - Index type affects search speed
   - HNSW fastest but more memory
   - IVF good balance for large sets

## Common Issues and Solutions

1. Dimension Mismatch
   ```yaml
   embed_config:
     use_pca: true      # Enable PCA
     dimension: 384     # Set target dimension
   ```

2. Memory Issues
   ```yaml
   embed_config:
     batch_size: 128    # Reduce batch size
     half_precision: true
   ```

3. Slow Processing
   ```yaml
   embed_config:
     num_workers: 8     # Increase workers
   vector_db:
     batch_size: 500    # Larger batches
   ```

4. Poor Accuracy
   ```yaml
   chunking:
     strategy: sentence  # Change chunking
     chunk_size: 500    # Smaller chunks
   retrieval:
     min_score: 0.7     # Higher threshold
   ```
