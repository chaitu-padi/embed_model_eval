# Retrieval for Milvus and Qdrant
import time
import numpy as np
from qdrant_client.http import models as qdrant_models
import logging

def retrieve_milvus(col, embeddings, top_k, metric_type, nprobe):
    search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}
    results = col.search([embeddings[0].tolist()], "embedding", search_params, limit=top_k)
    retrieved_ids = [hit.id for hit in results[0]] if results and len(results) > 0 else []
    return retrieved_ids

def retrieve_qdrant(client, collection, embeddings, top_k, semantic_query=None, embed_model=None):
    # If semantic_query is provided, use it for semantic search
    if semantic_query and embed_model:
        # Generate embedding for the query
        query_emb = embed_model.encode([semantic_query], show_progress_bar=False)[0]
        search_res = client.search(collection_name=collection, query_vector=query_emb.tolist(), limit=top_k, with_payload=True)
    else:
        # Default: use first embedding
        search_res = client.search(collection_name=collection, query_vector=embeddings[0].tolist(), limit=top_k, with_payload=True)
    retrieved_ids = [hit.id for hit in search_res]
    return retrieved_ids

def retrieve_qdrant_with_results(client, collection, semantic_query, embed_model, top_k,
                               config=None, payload_filter_cfg=None, query_embedding=None):
    """Enhanced retrieval with improved scoring, filtering and diversity
    
    Args:
        client: Qdrant client instance
        collection: Name of the collection to search
        semantic_query: The search query text
        embed_model: The embedding model to use
        top_k: Number of results to return
        config: Configuration dictionary from config.yaml
        payload_filter_cfg: Optional payload filters
        query_embedding: Pre-computed query embedding (optional)
    """
    logging.info(f"Starting retrieval for collection: {collection}")
    logging.info(f"Semantic query: {semantic_query}")
    logging.info(f"Top-k: {top_k}")
    try:
        # Get config or use defaults
        if config is None:
            config = {}
        
        retrieval_cfg = config.get('retrieval', {})
        embed_cfg = config.get('embed_config', {})
        
        # Get parameters from config
        score_threshold = retrieval_cfg.get('score_threshold', 0.6)  # Lowered threshold for better recall
        rerank_results = retrieval_cfg.get('rerank_results', True)
        diversity_weight = retrieval_cfg.get('diversity_weight', 0.3)
        normalize = embed_cfg.get('normalize', True)
        
        # Generate embedding if not provided
        if query_embedding is None:
            logging.info("Generating query embedding...")
            query_embedding = embed_model.encode(
                [semantic_query],
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )[0]
            logging.info(f"Query embedding shape: {query_embedding.shape}, norm: {np.linalg.norm(query_embedding):.4f}")
        
        # Get vector db config
        vector_db_cfg = config.get('vector_db', {})
        vector_index_cfg = vector_db_cfg.get('vector_index', {})
        
        # Get the metric type from config
        metric_type = retrieval_cfg.get('metric_type', 'COSINE')
        
        # Construct search parameters from config
        search_params = {
            "collection_name": collection,
            "query_vector": query_embedding.tolist(),
            "limit": top_k * 2 if rerank_results else top_k,
            "with_payload": True,
            "score_threshold": score_threshold
        }

        # Add scoring parameters from config with higher ef value for better recall
        ef_search = vector_index_cfg.get('params', {}).get('ef_construct', 256)  # Increased from 128
        search_params["search_params"] = qdrant_models.SearchParams(
            hnsw_ef=ef_search,
            exact=vector_index_cfg.get('type', 'hnsw').lower() == 'plain'
        )
        
        logging.info(f"Search parameters: score_threshold={score_threshold}, ef_search={ef_search}")
        logging.info(f"Using metric type: {metric_type}")
        
        # Add payload filter if configured
        if payload_filter_cfg:
            search_params["query_filter"] = payload_filter_cfg
            logging.info(f"Using payload filter: {payload_filter_cfg}")
            
        # Search with timing
        start_time = time.time()
        search_results = client.search(**search_params)
        query_time = time.time() - start_time
        
        logging.info(f"Found {len(search_results)} results in {query_time:.2f} seconds")
        
        # Extract results and scores
        results = []
        seen_content = set()  # Track unique content for diversity
        
        if rerank_results:
            # Rerank using both relevance and diversity
            candidates = []
            for hit in search_results:
                # Create fingerprint of content for diversity check
                content_key = str(sorted(hit.payload.items()))
                diversity_penalty = 0.0
                
                if content_key in seen_content:
                    diversity_penalty = diversity_weight
                
                # Combined score considering both relevance and diversity
                final_score = hit.score - diversity_penalty
                candidates.append((hit, final_score, content_key))
                
            # Sort by final score and take top_k
            candidates.sort(key=lambda x: x[1], reverse=True)
            for hit, score, content_key in candidates[:top_k]:
                seen_content.add(content_key)
                logging.info(f"Match score: {score:.4f} for document with payload: {hit.payload}")
                results.append((hit.id, hit.payload))
        else:
            # Direct results without reranking
            for hit in search_results[:top_k]:
                logging.info(f"Match score: {hit.score:.4f} for document with payload: {hit.payload}")
                results.append((hit.id, hit.payload))
        
        metrics = {
            'avg_query_time': query_time,
            'query_throughput': 1.0 / query_time if query_time > 0 else 0,
            'total_query_time': query_time,
            'scores': [hit.score for hit in search_results[:top_k]],  # Track similarity scores
            'diversity_score': len(seen_content) / len(results) if results else 0  # Measure result diversity
        }
        
        return results, metrics
    except Exception as e:
        logging.error(f"Retrieval error: {str(e)}")
        raise
