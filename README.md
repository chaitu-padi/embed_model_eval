# Embedding Model Evaluation (Work In Progress)

This project provides a modular framework for evaluating embedding model performance across multiple vector databases (Milvus, Qdrant, etc.) and data sources (CSV, Oracle, files, RDBMS). It supports advanced configuration, chunking, semantic/dense/sparse embeddings, and both CLI and UI interfaces.

## Features
- **Modular codebase**: Organized into data sources, embedding models, vector databases, retrieval, evaluation, reporting, config, and utilities.
- **Configurable pipeline**: All parameters (chunking, parallelism, indexing, batch size, etc.) are set via `config/config.yaml`.
- **Data sources**: Supports CSV, Oracle, JSON, TXT, and more. Selective column embedding or full file embedding.
- **Chunking strategies**: None, sentence, sliding window, fixed length.
- **Embedding models**: Semantic, dense, sparse, hybrid. Model and dimension are configurable.
- **Vector DBs**: Milvus (local/Docker), Qdrant (local server). Batch size and upsert retry logic are configurable.
- **Semantic search**: Qdrant retrieval supports semantic queries from config.
- **Evaluation metrics**: Accuracy, recall, precision, F1.
- **Progress bars**: For embedding and DB ingestion using `tqdm`.
- **Logging**: All steps use Python logging for robust error handling and progress.
- **Reporting**: Generates a detailed HTML report (`embedding_report.html`) with model, config, resource usage, timings, and metrics for easy comparison.
- **.gitignore**: Excludes cache, virtual env, data, and report files.

## Usage
### CLI (Backend)
The CLI pipeline is fully runnable:
```bash
python main.py --config config/config.yaml
```
- All configuration is loaded from `config/config.yaml`.
- Embedding, ingestion, retrieval, and evaluation are performed end-to-end.
- HTML report is generated at the end.

### Streamlit UI (Frontend)
- The Streamlit UI (`app.py`) is **in progress** and not yet feature-complete.
- Backend logic is stable and can be used for all evaluations.

## Configuration
See `config/config.yaml` for all options:
- Data source type, file, and columns
- Embedding model, batch size, dimension, parallelism
- Chunking strategy and parameters
- Vector DB type, host, port, collection, batch size, upsert retries
- Retrieval parameters, including semantic query
- Evaluation metrics and ground truth

## Current Status
- **Work in Progress**: Backend CLI is stable and runnable. Streamlit UI is under development.
- Latest features: semantic search for Qdrant, upsert retry logic, batch size config, selective column embedding, robust HTML reporting.
- See `.gitignore` for excluded files and folders.

## How to Contribute
- Clone the repo and run the CLI pipeline.
- Open issues or PRs for new features, bug fixes, or UI improvements.

## License
MIT
