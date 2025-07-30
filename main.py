import argparse
import logging
import os
import datetime
import numpy as np
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
        format='%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s'
    )

def run_pipeline(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Execute the main embedding evaluation pipeline for a specific model"""
    debug_retrieval_only = config.get('debug_retrieval_only', False)
    vdb_cfg = config['vector_db'].copy()  # Create a copy to avoid modifying original
    emb_cfg = config['embed_config']
    
    # Get base collection name from config and dimension
    base_collection = config['vector_db'].get('collection', 'embeddings')  # Get from original config
    dimension = emb_cfg.get('dimension', 384)
    
    # Create collection name with exact model name and dimension
    vdb_cfg['collection'] = f"{base_collection}_{model_name}_{dimension}"
    logging.info(f"Created collection name: {vdb_cfg['collection']}")
    

    """
    # Clean up existing collection before use
    if not debug_retrieval_only:
        from qdrant_client import QdrantClient
        db_client = QdrantClient(
            host=vdb_cfg.get('host', 'localhost'),
            port=int(vdb_cfg.get('port', 6333))
        )
        try:
            db_client.delete_collection(vdb_cfg['collection'])
            logging.info(f"[Setup] Cleaned up existing collection: {vdb_cfg['collection']}")
        except Exception as e:
            logging.info(f"[Setup] No existing collection to clean: {e}")
    """
    # Validate PCA settings when using multiple models
    if model_name in emb_cfg.get('models', []):
        if not emb_cfg.get('use_pca', False):
            logging.warning(
                f"Multiple models configured but PCA is disabled. "
                f"This may cause dimension mismatch errors if models output different dimensions. "
                f"Enable PCA in config.yaml with use_pca: true"
            )
    if debug_retrieval_only:
        logging.info("[DEBUG] Retrieval-only mode enabled. Skipping embedding generation and insertion.")
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        db_client = QdrantClient(
            host=vdb_cfg.get('host', 'localhost'),
            port=int(vdb_cfg.get('port', 6333))
        )
        # Get the collection info to get total vector count
        collection_info = db_client.get_collection(vdb_cfg['collection'])
        total_embeddings = collection_info.points_count
        logging.info(f"[DEBUG] Found {total_embeddings} existing embeddings in collection {vdb_cfg['collection']}")
        
        emb_cfg = config['embed_config']
        model = SentenceTransformer(emb_cfg.get('model', 'all-MiniLM-L6-v2'))
        embeddings = None  # Set to None instead of empty list to indicate no new embeddings
        data_load_time = 0
        embedding_time = 0
        insertion_time = 0
    else:
        # 1. Load data
        logging.info("[Step 1 start] Data Source Loading...")
        import time
        t0 = time.time()
        df, collection_name, texts, payloads = load_data(config)
        data_load_time = time.time() - t0
        logging.info(f"[Step 1 end] Loaded {len(df)} rows. Time taken: {data_load_time:.2f} seconds.")

        # 2. Generate embeddings 
        logging.info("[Step 2 start] Generating Embeddings ...")
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
            

        embeddings, model_metrics = generate_embeddings(
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
        logging.info(f"[Step 2 metrics] Model load time: {model_metrics['model_load_time']:.2f}s, "
                    f"Memory usage: {model_metrics['model_memory_mb']:.2f}MB, "
                    f"GPU memory: {model_metrics['gpu_memory_used']:.2f}GB")


        # Use actual embedding dimension for DB operations and store it
        actual_vector_size = embeddings.shape[1]
        # Store the actual dimension in config for retrieval
        emb_cfg['actual_dimension'] = actual_vector_size
        logging.info(f"Actual vector dimension after processing: {actual_vector_size}")
        
        if actual_vector_size != emb_cfg.get('dimension', 384):
            logging.info(f"Note: Final dimension ({actual_vector_size}) differs from target ({emb_cfg.get('dimension', 384)})")
            if use_pca:
                logging.info("This is due to PCA auto-adjustment based on data characteristics")

        # 3. Insert into vector database (insert_embeddings_qdrant will ensure collection exists with correct dimension)
        from vector_databases.insertion import insert_embeddings_qdrant, setup_qdrant_indexing
        logging.info("[Step 3 start] Vector Database Insertion...")
        t2 = time.time()
        db_client, db_metrics = insert_embeddings_qdrant(
            embeddings,
            texts,
            payloads,
            collection_name=vdb_cfg.get('collection', 'embeddings'),
            vector_size=actual_vector_size,
            host=vdb_cfg.get('host', 'localhost'),
            port=int(vdb_cfg.get('port', 6333)),
            batch_size=int(vdb_cfg.get('batch_size', 100))
        )
        insertion_time = db_metrics['insertion_time']
        logging.info(f"[Step 3 end] Inserted embeddings into vector DB. Time taken: {insertion_time:.2f} seconds. Insert rate: {db_metrics['insert_rate']:.2f} vectors/second.")

        # 4. Configure indexing (if Qdrant)
        if vdb_cfg.get('type', '').lower() == 'qdrant':
            logging.info("[Step 4 start] Configuring Qdrant indexing...")
            setup_qdrant_indexing(
                db_client,
                vdb_cfg.get('collection', 'embeddings'),
                vdb_cfg.get('vector_index', {}),
                vdb_cfg.get('payload_index', [])
            )
            logging.info("[Step 4 end] Qdrant indexing configured.")

    # Initialize metrics dictionary
    metrics = {}
    
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
                from embeddings.generator import generate_embeddings, get_model_dimension  # Import generator for PCA
                emb_cfg = config['embed_config']
                model_name = emb_cfg.get('model', 'all-MiniLM-L6-v2')
                embed_model = SentenceTransformer(model_name)
                
                # Get model's output dimension and configure PCA
                model_output_dim = get_model_dimension(model_name)
                target_dim = emb_cfg.get('dimension', 384)
                use_pca = emb_cfg.get('use_pca', False)
                
                # Check actual dimension from vector DB collection
                try:
                    collection_info = db_client.get_collection(collection_name)
                    actual_dim = collection_info.config.params.vectors.size
                    logging.info(f"Collection vector size: {actual_dim}")
                except Exception as e:
                    logging.warning(f"Could not get collection dimension: {e}")
                    # Fall back to target dimension from config
                    actual_dim = target_dim
                    logging.info(f"Using target dimension from config: {actual_dim}")
                
                # Only use PCA if we need dimension reduction and dimensions don't match
                if model_output_dim == actual_dim:
                    use_pca = False
                    logging.info(f"Model output dimension ({model_output_dim}) matches collection dimension. Skipping PCA.")
                
                logging.info(f"Model dimension: {model_output_dim}, Collection dimension: {actual_dim}, Use PCA: {use_pca}")
                
                # Load PCA model if needed
                pca = None
                pca_config = emb_cfg.get('pca_config', {})
                if use_pca:
                    logging.info(f"PCA config: {pca_config}")
                    from embeddings.generator import load_pca_model
                    # Use actual_dim from collection instead of target_dim
                    pca = load_pca_model(actual_dim, pca_config)
                    if pca is not None:
                        logging.info(f"Loaded PCA model expects input dimension: {pca.n_features_in_} and outputs: {pca.n_components_}")
                    else:
                        logging.error(f"Required PCA model for dimension {actual_dim} not found. Cannot proceed with retrieval.")
                        raise ValueError(f"PCA model not found for dimension {actual_dim}")
                
            if semantic_query:
                logging.info(f"[Step 5] Performing semantic search: {semantic_query}")
                # Generate query embedding with proper dimension
                if not debug_retrieval_only:
                    # Generate query embedding with proper dimension
                    query_embedding = embed_model.encode(
                        [semantic_query],
                        normalize_embeddings=False,  # Don't normalize yet
                        convert_to_numpy=True
                    )
                    
                    logging.info(f"Initial query embedding shape: {query_embedding.shape}")
                    
                    # Ensure query_embedding is 2D and has right shape
                    if len(query_embedding.shape) == 3:
                        query_embedding = query_embedding.reshape(query_embedding.shape[0], -1)
                    elif len(query_embedding.shape) == 1:
                        query_embedding = query_embedding.reshape(1, -1)
                    
                    logging.info(f"Query embedding shape: {query_embedding.shape}")
                    
                    # Only apply PCA if needed (dimensions don't match and PCA is enabled)
                    if use_pca and pca is not None:
                        if query_embedding.shape[-1] != pca.n_features_in_:
                            logging.error(f"Model output dimension ({query_embedding.shape[-1]}) doesn't match "
                                      f"PCA input dimension ({pca.n_features_in_})")
                            raise ValueError("Model output dimension doesn't match PCA input dimension. "
                                          "Make sure you're using the same model that was used for training.")
                            
                        # Transform the query embedding using PCA
                        query_embedding = pca.transform(query_embedding)
                        # Take first embedding since we only have one query
                        query_embedding = query_embedding[0]
                        logging.info(f"Query embedding shape after PCA: {query_embedding.shape}")
                    else:
                        # If not using PCA, just take the first embedding
                        query_embedding = query_embedding[0]
                    
                    # Apply normalization after all dimension adjustments
                    if emb_cfg.get('normalize', True):
                        query_embedding = query_embedding / np.linalg.norm(query_embedding)
                    
                    results, query_metrics = retrieve_qdrant_with_results(
                        db_client,
                        collection_name,
                        semantic_query,
                        embed_model,
                        top_k,
                        config=config,
                        payload_filter_cfg=payload_filter_cfg,
                        query_embedding=query_embedding
                    )
                else:
                    results, query_metrics = retrieve_qdrant_with_results(
                        db_client,
                        collection_name,
                        semantic_query,
                        embed_model,
                        top_k,
                        config=config,
                        payload_filter_cfg=payload_filter_cfg
                    )
                
                # Update metrics with query performance data
                metrics.update({
                    'avg_query_time': query_metrics['avg_query_time'],
                    'query_throughput': query_metrics['query_throughput'],
                    'total_query_time': query_metrics['total_query_time']
                })
                
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
    
    # Ensure we have relevant IDs for evaluation
    relevant_ids = eval_cfg.get('relevant_ids', [])
    top_k = ret_cfg.get('top_k', 5)
    
    if not relevant_ids:
        logging.warning("[Step 6] No relevant IDs provided for evaluation. Metrics will be zero.")
        
    # Calculate evaluation metrics
    eval_metrics = evaluate(retrieved_ids, relevant_ids, top_k)
    logging.info(f"[Step 6] Evaluation Metrics: Precision={eval_metrics['precision']:.3f}, "
                f"Recall={eval_metrics['recall']:.3f}, F1={eval_metrics['f1']:.3f}")
    
    # Calculate rates
    embeddings_per_second = (len(embeddings) / embedding_time if 'embeddings' in locals() 
                           and embeddings is not None and embedding_time > 0 else 0)
    
    processing_rate = (len(texts) / data_load_time if 'texts' in locals() 
                      and texts is not None and data_load_time > 0 else 0)
        
    # Initialize metrics with all values
    metrics = {
        # Timing metrics
        'data_load_time': data_load_time,
        'embedding_time': embedding_time,
        'insertion_time': insertion_time,
        'retrieval_time': retrieval_time if 'retrieval_time' in locals() else 0.0,
        'model_load_time': model_metrics['model_load_time'] if 'model_metrics' in locals() else 0.0,
        
        # Resource metrics
        'model_memory_mb': model_metrics['model_memory_mb'] if 'model_metrics' in locals() else 0.0,
        'gpu_memory_used': model_metrics['gpu_memory_used'] if 'model_metrics' in locals() else 0.0,
        'device': model_metrics['device'] if 'model_metrics' in locals() else 'cpu',
        'index_build_time': db_metrics.get('index_build_time', 0.0) if 'db_metrics' in locals() else 0.0,
        'insert_rate': db_metrics.get('insert_rate', 0.0) if 'db_metrics' in locals() else 0.0,
        
        # Performance metrics
        'processing_rate': f"{processing_rate:.2f}" if 'processing_rate' in locals() else '0.00',
        'embeddings_per_second': f"{embeddings_per_second:.2f}" if 'embeddings_per_second' in locals() else '0.00',
        'total_embeddings': (
            total_embeddings if 'total_embeddings' in locals()
            else (len(embeddings) if 'embeddings' in locals() and embeddings is not None else 0)
        ),
        
        # Embedding configuration
        'embedding_batch_size': emb_cfg.get('batch_size', 'N/A'),
        'embedding_dimension': emb_cfg.get('dimension', 'N/A'),
        'normalize': emb_cfg.get('normalize', True),
        'use_pca': emb_cfg.get('use_pca', False),
        
        # Vector DB configuration
        'db_type': vdb_cfg.get('type', 'qdrant'),
        'db_batch_size': vdb_cfg.get('batch_size', 'N/A'),
        'db_upsert_retries': vdb_cfg.get('upsert_retries', 'N/A'),
        'db_retry_delay': vdb_cfg.get('upsert_retry_delay', 'N/A'),
        'vector_index': vdb_cfg.get('vector_index', {}),
        'payload_index': vdb_cfg.get('payload_index', []),
        
        # Retrieval configuration
        'semantic_query': ret_cfg.get('semantic_query', 'N/A') if 'ret_cfg' in locals() else 'N/A',
        'top_k': ret_cfg.get('top_k', 5) if 'ret_cfg' in locals() else 5,
        'filter_applied': bool(payload_filter_cfg) if 'payload_filter_cfg' in locals() else False,
        
        # Evaluation metrics
        'precision': eval_metrics.get('precision', 0.0),
        'recall': eval_metrics.get('recall', 0.0),
        'f1_score': eval_metrics.get('f1', 0.0)
    }
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
        logging.info(f"[Process] Evaluating model: {model_name}")
        
        # Create a deep copy of config for this model
        model_config = {
            **config.copy(),
            'vector_db': config['vector_db'].copy(),  # Ensure clean copy of vector_db config
            'embed_config': config['embed_config'].copy()  # Ensure clean copy of embed_config
        }
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
