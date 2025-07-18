# Retrieval for Milvus and Qdrant

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
