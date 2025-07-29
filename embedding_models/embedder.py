
# Embedding model loader and batch embedder
import time
import psutil
import torch
from concurrent.futures import ThreadPoolExecutor

def get_supported_models():
    """
    Returns a dictionary of supported embedding models and their loader functions.
    Extend this to support custom, Cohere, Jina, Nomic, etc.
    """
    return {
        # Sentence Transformers
        "sentence-transformers_all-mpnet-base-v2": lambda: __import__('sentence_transformers').sentence_transformers.SentenceTransformer('all-mpnet-base-v2'),
        "sentence-transformers_all_miniLM-L6-v2": lambda: __import__('sentence_transformers').sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2'),
        "sentence-transformers_paraphrase-distilroberta-base-v1": lambda: __import__('sentence_transformers').sentence_transformers.SentenceTransformer('paraphrase-distilroberta-base-v1'),
        # Custom/other models (add actual loader logic as needed)
        "vprelovac_universe-sentence-encoder_4": lambda: NotImplementedError("Add loader for vprelovac_universe-sentence-encoder_4"),
        "jinaai_jina-embeddings-v2-base-en": lambda: NotImplementedError("Add loader for jinaai_jina-embeddings-v2-base-en"),
        "intfloat_e5-base-v2": lambda: __import__('sentence_transformers').sentence_transformers.SentenceTransformer('intfloat/e5-base-v2'),
        "intfloat_e5-base": lambda: __import__('sentence_transformers').sentence_transformers.SentenceTransformer('intfloat/e5-base'),
        "nomic-ai_nomic-embed-text-v1.5": lambda: NotImplementedError("Add loader for nomic-ai_nomic-embed-text-v1.5"),
        "Cohere_cohere-embed-english-v3.0": lambda: NotImplementedError("Add loader for Cohere_cohere-embed-english-v3.0"),
        "Cohere_cohere-embed-english-light-v3.0": lambda: NotImplementedError("Add loader for Cohere_cohere-embed-english-light-v3.0"),
        #"Cohere_cohere-embed-multilingual-v3.0": lambda: NotImplementedError("Add loader for Cohere_cohere-embed-multilingual-v3.0"),
        "BAAI_bge-large-en-v1.5": lambda: __import__('sentence_transformers').sentence_transformers.SentenceTransformer('BAAI/bge-large-en-v1.5'),
        "BAAI_bge-base-en-v1.5": lambda: __import__('sentence_transformers').sentence_transformers.SentenceTransformer('BAAI/bge-base-en-v1.5'),
        "nomic-ai_nomic-bert-2048": lambda: NotImplementedError("Add loader for nomic-ai_nomic-bert-2048"),
    }

# Global cache for models
_model_cache = {}

def load_embedding_model(model_name):
    """
    Loads the embedding model by name and tracks resource usage.
    Returns tuple of (model, metrics_dict).
    Uses a cache to avoid reloading models.
    """
    global _model_cache
    if model_name in _model_cache:
        return _model_cache[model_name]
        
    # Initial memory state
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    
    start_time = time.time()
    
    models = get_supported_models()
    try:
        if model_name in models:
            loader = models[model_name]
            try:
                model = loader()
                if isinstance(model, NotImplementedError):
                    raise model
            except Exception as e:
                raise RuntimeError(f"Model '{model_name}' not supported or failed to load: {e}")
        else:
            # Fallback to sentence-transformers
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        
        # Calculate resource usage
        end_time = time.time()
        current_memory = process.memory_info().rss / (1024 * 1024)
        current_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        
        metrics = {
            'model_load_time': end_time - start_time,
            'model_memory_mb': current_memory - initial_memory,
            'gpu_memory_used': (current_gpu_memory - initial_gpu_memory) / 1024,  # Convert to GB
            'device': str(next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu')
        }
        
        # Cache the model and metrics
        _model_cache[model_name] = (model, metrics)
        return model, metrics
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

def embed_texts(texts, emb_cfg):
    """
    Embeds texts using the selected model and configuration.
    Supports batch and parallel embedding.
    Returns tuple of (embeddings, model_metrics).
    """
    model_name = emb_cfg.get('model', 'sentence-transformers_all_miniLM-L6-v2')
    batch_size = int(emb_cfg.get('batch_size', 64))
    parallelism = int(emb_cfg.get('parallelism', 1))
    normalize = emb_cfg.get('normalize', True)
    
    model, model_metrics = load_embedding_model(model_name)

    # For single-threaded execution, show progress bar
    if parallelism <= 1:
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=normalize
        )
        return embeddings, model_metrics
        
    # For multi-threaded execution, split into batches
    def embed_batch(batch):
        return model.encode(
            batch,
            show_progress_bar=False,
            batch_size=batch_size,
            normalize_embeddings=normalize
        )

    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    embeddings = []
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        results = list(executor.map(embed_batch, batches))
    for batch_emb in results:
        embeddings.extend(batch_emb)
    
    # Convert embeddings to numpy array
    import numpy as np
    embeddings = np.array(embeddings)
    return embeddings, model_metrics
