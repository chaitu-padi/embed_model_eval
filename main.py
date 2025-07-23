import argparse
import logging
import os
import datetime
from typing import Dict, Any

from config.config_loader import ConfigLoader
from pipeline.embedding_pipeline import EmbeddingPipeline
from data_sources.loader import load_data
from evaluation.metrics import evaluate
from reporting.report import print_report

def setup_logging() -> None:
    """Configure logging format and level"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

def run_pipeline(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Execute the main embedding evaluation pipeline for a specific model"""
    debug_retrieval_only = config.get('debug_retrieval_only', False)
    vdb_cfg = config['vector_db']
    
    # Update the collection name to include model name to keep results separate
    model_safe_name = model_name.replace('/', '_').replace('-', '_')
    vdb_cfg['collection'] = f"{vdb_cfg.get('collection', 'embeddings')}_{model_safe_name}"
    if debug_retrieval_only:
        logging.info("[DEBUG] Retrieval-only mode enabled. Skipping embedding generation and insertion.")
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        db_client = QdrantClient(
            host=vdb_cfg.get('host', 'localhost'),
            port=int(vdb_cfg.get('port', 6333))
        )
        emb_cfg = config['embed_config']
        model = SentenceTransformer(emb_cfg.get('model', 'all-MiniLM-L6-v2'))
        embeddings = []
        data_load_time = 0
        embedding_time = 0
        insertion_time = 0
    else:
        # 1. Load data
        logging.info("\n[Step 1 start] Data Source Loading...")
        import time
        t0 = time.time()
        df, collection_name, texts, payloads = load_data(config)
        data_load_time = time.time() - t0
        logging.info(f"[Step 1 end] Loaded {len(df)} rows. Time taken: {data_load_time:.2f} seconds.")

        # 2. Generate embeddings 
        logging.info("\n[Step 2 start] Generating Embeddings ...")
        from embeddings.generator import generate_embeddings
        emb_cfg = config['embed_config']
        t1 = time.time()
        
        # Get all required parameters from config
        if 'model' not in emb_cfg:
            raise ValueError("model name must be specified in embed_config")
        if 'dimension' not in emb_cfg:
            raise ValueError("dimension must be specified in embed_config")
        if 'batch_size' not in emb_cfg:
            raise ValueError("batch_size must be specified in embed_config")
        if 'normalize' not in emb_cfg:
            raise ValueError("normalize must be specified in embed_config")
        if 'use_pca' not in emb_cfg:
            raise ValueError("use_pca must be specified in embed_config")
            
        embeddings = generate_embeddings(
            texts=texts,
            model_name=emb_cfg['model'],
            batch_size=int(emb_cfg['batch_size']),
            normalize=emb_cfg['normalize'],
            target_dim=int(emb_cfg['dimension']),
            use_pca=emb_cfg['use_pca'],
            pca_config=emb_cfg.get('pca_config', {})  # pca_config is optional
        )
        embedding_time = time.time() - t1
        logging.info(f"[Step 2 end] Generated {len(embeddings)} embeddings. Time taken: {embedding_time:.2f} seconds.")

        # 3. Insert into vector database
        logging.info("[Step 3 start] Vector Database Insertion...")
        from vector_databases.insertion import insert_embeddings_qdrant, setup_qdrant_indexing
        t2 = time.time()
        db_client, insertion_time = insert_embeddings_qdrant(
            embeddings,
            texts,
            payloads,
            collection_name=vdb_cfg.get('collection', 'embeddings'),
            vector_size=emb_cfg.get('dimension', 768),
            host=vdb_cfg.get('host', 'localhost'),
            port=int(vdb_cfg.get('port', 6333)),
            batch_size=int(vdb_cfg.get('batch_size', 100))
        )
        insertion_time = time.time() - t2
        # Print this log after insertion
        logging.info(f"[Step 3 end] Inserted embeddings into vector DB. Time taken: {insertion_time:.2f} seconds.")

        # 4. Configure indexing (if Qdrant)
        if vdb_cfg.get('type', '').lower() == 'qdrant':
            logging.info("\n[Step 4 start] Configuring Qdrant indexing...")
            setup_qdrant_indexing(
                db_client,
                vdb_cfg.get('collection', 'embeddings'),
                vdb_cfg.get('vector_index', {}),
                vdb_cfg.get('payload_index', [])
            )
            logging.info("\n[Step 4 end] Qdrant indexing configured.")

    # 5. Retrieval
    logging.info("[Step 5 start] Qdrant retrieval...")
    t3 = time.time()  # Start timing retrieval
    collection_name = vdb_cfg.get('collection', 'embeddings')
    logging.info(f"Retrieving from collection: {collection_name}")
    logging.info(f"[DEBUG] Vector DB Index config: {vdb_cfg.get('vector_index', {})}")
    logging.info(f"[DEBUG] Payload index config: {vdb_cfg.get('payload_index', [])}")
    ret_cfg = config.get('retrieval', {})
    retrieved_ids = []
    retrieval_time = 0  # Initialize retrieval time
    if vdb_cfg.get('type', '').lower() == 'qdrant':
        try:
            from retrieval.retriever import retrieve_qdrant_with_results
            semantic_query = ret_cfg.get('semantic_query', None)
            payload_filter_cfg = ret_cfg.get('payload_filter', None)
            top_k = int(ret_cfg.get('top_k', 5))
            # Always use a valid model for semantic search
            if debug_retrieval_only:
                embed_model = model
            else:
                from sentence_transformers import SentenceTransformer
                emb_cfg = config['embed_config']
                embed_model = SentenceTransformer(emb_cfg.get('model', 'all-MiniLM-L6-v2'))
            if semantic_query:
                logging.info(f"[Step 5] Performing semantic search: {semantic_query}")
                results = retrieve_qdrant_with_results(db_client, collection_name, semantic_query, embed_model, top_k, payload_filter_cfg)
                for idx, (rid, payload) in enumerate(results):
                    payload_str = ', '.join([f"{i}:{k}={v}" for i, (k, v) in enumerate(payload.items())]) if payload else 'None'
                    logging.info(f"[Step 5][Result {idx}] ID: {rid}, Payload: {payload_str}")
                    retrieved_ids.append(rid)
                retrieval_time = time.time() - t3  # Calculate total retrieval time
                logging.info(f"[Step 5] Retrieval time: {retrieval_time:.2f} seconds")
            else:
                logging.info("[Step 5] No semantic query provided. Skipping retrieval.")
        except Exception as e:
            logging.error(f"[Step 5] Retrieval failed: {e}")
    else:
        logging.info("[Step 5] Retrieval not implemented for this DB type.")

    # 6. Evaluate and collect metrics
    eval_cfg = config.get('evaluation', {})
    logging.info(f"[Step 6] Evaluation: Retrieved IDs: {retrieved_ids}")
    logging.info(f"[Step 6] Evaluation: Relevant IDs: {eval_cfg.get('relevant_ids', [])}")
    metrics = evaluate(retrieved_ids, eval_cfg.get('relevant_ids', []), ret_cfg.get('top_k', 5))
    # Get batch size and retries from config
    batch_size = vdb_cfg.get('batch_size', 'N/A')
    upsert_retries = vdb_cfg.get('upsert_retries', 'N/A')
    
    # Calculate embeddings per second
    if embedding_time > 0 and len(embeddings) > 0:
        embeddings_per_second = len(embeddings) / embedding_time
    else:
        embeddings_per_second = 0
        
    # Calculate processing rate
    if data_load_time > 0 and len(texts) > 0:
        processing_rate = len(texts) / data_load_time
    else:
        processing_rate = 0
    
    metrics.update({
        # Timing metrics
        'data_load_time': data_load_time,
        'embedding_time': embedding_time,
        'insertion_time': insertion_time,
        'retrieval_time': retrieval_time if 'retrieval_time' in locals() else 0,
        
        # Performance rates
        'processing_rate': f"{processing_rate:.2f}",
        'embeddings_per_second': f"{embeddings_per_second:.2f}",
        'total_embeddings': len(embeddings),
        
        # Configuration
        'batch_size': batch_size,
        'upsert_retries': upsert_retries
    })
    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding Model Evaluation CLI")
    parser.add_argument(
        '--config', 
        type=str,
        default='config/config.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()

    setup_logging()
    logging.info("[Process] Starting Embedding Model Evaluation Pipeline...")

    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.config

    # Get list of models to evaluate
    embed_cfg = config.get('embed_config', {})
    models_to_evaluate = embed_cfg.get('models', [embed_cfg.get('default_model', 'all-MiniLM-L6-v2')])
    
    # Store metrics for each model
    all_metrics = {}
    
    # Run pipeline for each model
    for model_name in models_to_evaluate:
        logging.info(f"\n[Process] Evaluating model: {model_name}")
        
        # Create a copy of config for this model
        model_config = config.copy()
        model_config['embed_config']['model'] = model_name
        
        # Run pipeline and get metrics
        metrics = run_pipeline(model_config, model_name)
        all_metrics[model_name] = metrics
        
        # Extract model config values
        model_dim = embed_cfg.get('dimension', 'N/A')
        batch_size = embed_cfg.get('batch_size', 'N/A')
        parallelism = embed_cfg.get('parallelism', 'N/A')
    
    # Generate report with all models
    now = datetime.datetime.now()
    date_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    report_filename = f"model_evaluation_report_{date_str}"
    
    print_report(
        models_to_evaluate,
        config['data_source'],
        config['vector_db'].get('type', '').lower(),
        config['vector_db'].get('host', 'localhost'),
        config['vector_db'].get('port', '19530'),
        config['vector_db'].get('collection', 'embeddings'),
        all_metrics,
        config['retrieval'].get('top_k', 5),
        model_dim,
        batch_size,
        parallelism,
        report_filename=report_filename,
        config=config  # Pass the full config
    )

    logging.info("\n[Process] Embedding Model Evaluation Pipeline Complete.")

if __name__ == "__main__":
    main()
