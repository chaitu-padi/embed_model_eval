# Retrieval for Milvus and Qdrant
from qdrant_client.http import models as qdrant_models

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

def retrieve_qdrant_with_results(client, collection, semantic_query, embed_model, top_k, payload_filter_cfg=None, query_embedding=None):
    """
    Retrieve from Qdrant using semantic query, returning both IDs and payloads.
    
    Args:
        client: Qdrant client instance
        collection: Collection name
        semantic_query: Query text
        embed_model: SentenceTransformer model
        top_k: Number of results to return
        payload_filter_cfg: Filter configuration for payloads
        query_embedding: Pre-computed query embedding (optional)
    """
    # Use provided query embedding or generate new one
    query_emb = query_embedding if query_embedding is not None else embed_model.encode([semantic_query], show_progress_bar=False)[0]
    must_conditions = []
    if payload_filter_cfg:
        for key, val in payload_filter_cfg.items():
            if isinstance(val, dict) and ('gte' in val or 'lte' in val):
                must_conditions.append(qdrant_models.FieldCondition(
                    key=key,
                    range=qdrant_models.Range(
                        gte=val.get('gte'),
                        lte=val.get('lte')
                    )
                ))
            else:
                must_conditions.append(qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchValue(value=val)
                ))
    payload_filter = qdrant_models.Filter(must=must_conditions) if must_conditions else None

    # Sorting by payload fields is not supported in this Qdrant client version
    search_res = client.search(
        collection_name=collection,
        query_vector=query_emb.tolist(),
        limit=top_k,
        with_payload=True,
        query_filter=payload_filter
    )
    results = [(hit.id, hit.payload) for hit in search_res]
    return results
