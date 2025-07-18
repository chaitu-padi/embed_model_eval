# Embedding Model Evaluation

This project provides a modular framework to evaluate embedding performance across multiple data sources and vector databases. It supports:
- Parallel embedding
- Multiple chunking methodologies
- Semantic, dense, and sparse embeddings
- Semantic search support
- Parameterized evaluation for all combinations
- Configurable via Streamlit UI or CLI
- Saving/loading configurations for reproducibility

## Supported Data Sources
- Oracle, MySQL, MSSQL (RDBMS)
- CSV, JSON, TXT files
- Unstructured files (future extension)

## Supported Vector Databases
- Milvus (fully implemented)
- Redis, Qdrant, ChromaDB (extensible)

## Getting Started
1. Install dependencies from `requirements.txt`.
2. Start Milvus (see [Milvus Quickstart](https://milvus.io/docs/quickstart.md)).
3. Run the Streamlit UI:
   ```
   streamlit run app.py
   ```
   Or run via CLI:
   ```
   python main.py --source file --file yourdata.csv --model all-MiniLM-L6-v2 --milvus_host localhost --milvus_port 19530
   ```

## Project Structure
- `main.py`: CLI entry point for embedding and evaluation
- `app.py`: Streamlit UI for interactive configuration and evaluation
- `config_manager.py`: Save/load configurations
- `embedding/`: Embedding and chunking implementations
- `evaluation/`: Evaluation logic and metrics
- `configurations/`: Saved user configurations

## How It Works
1. Choose data source and vector DB in the UI or CLI.
2. Configure embedding parameters and save configurations.
3. The system checks/creates the Milvus collection (named after table or file).
4. Embeddings are generated and stored in the vector DB.
5. Retrieval and evaluation metrics (accuracy, recall, precision, F1, timing) are shown.

## Extending the Project
- Add support for other vector DBs by implementing connection and CRUD logic.
- Add more chunking and embedding models as needed.
- Extend data source support for unstructured files and other RDBMS.

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## License
MIT
