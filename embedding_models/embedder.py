
# Embedding model loader and batch embedder
import time
import psutil
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import tensorflow_hub as hub
import os
import logging
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModel, AutoTokenizer

class UniversalSentenceEncoder:
    def __init__(self):
        # Load model from TensorFlow Hub with error handling
        try:
            # First try to load from a cache directory to avoid downloading every time
            cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['TFHUB_CACHE_DIR'] = cache_dir
            
            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            self.device = "cpu"  # TensorFlow handles device placement
            self.dimension = 512  # USE v4 has fixed 512-dimensional output
        except Exception as e:
            raise RuntimeError(f"Failed to load Universal Sentence Encoder: {str(e)}")

    def encode(self, sentences, batch_size=32, show_progress_bar=True, normalize_embeddings=True):
        if isinstance(sentences, str):
            sentences = [sentences]

        # Process in batches
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_embeddings = self.model(batch).numpy()
            if normalize_embeddings:
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def parameters(self):
        # Dummy parameters for device checking
        return [torch.nn.Parameter(torch.zeros(1))]

"""
class CohereEmbedder:
    def __init__(self, model="embed-english-v3.0"):
        self.api_key = os.getenv('COHERE_API_KEY')
        if not self.api_key:
            raise ValueError("COHERE_API_KEY environment variable must be set")
        self.co = cohere.Client(self.api_key)
        self.model = model
        self.device = "cpu"  # Cohere is API-based

    def encode(self, sentences, batch_size=32, show_progress_bar=True, normalize_embeddings=True):
        if isinstance(sentences, str):
            sentences = [sentences]

        # Process in batches
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            response = self.co.embed(texts=batch, model=self.model)
            batch_embeddings = np.array(response.embeddings)
            if normalize_embeddings:
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def parameters(self):
        # Dummy parameters for device checking
        return [torch.nn.Parameter(torch.zeros(1))]
"""


def get_supported_models():
    """
    Returns a dictionary of supported embedding models and their loader functions.
    Extend this to support custom, Cohere, Jina, Nomic, etc.
    """
    return {
        # Sentence Transformers and Cross Encoders
        "cross-encoder/ms-marco-MiniLM-L-6-v2": lambda: __import__('sentence_transformers').CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'),
        "sentence-transformers_all-mpnet-base-v2": lambda: __import__('sentence_transformers').SentenceTransformer('all-mpnet-base-v2'),
        "sentence-transformers_all_miniLM-L6-v2": lambda: __import__('sentence_transformers').SentenceTransformer('all-MiniLM-L6-v2'),
        "sentence-transformers_paraphrase-distilroberta-base-v1": lambda: __import__('sentence_transformers').SentenceTransformer('paraphrase-distilroberta-base-v1'),
        # Custom/other models with HuggingFace/SentenceTransformers compatibility
        "vprelovac_universe-sentence-encoder_4": lambda: UniversalSentenceEncoder(),
        "jinaai_jina-embeddings-v2-base-en": lambda: __import__('sentence_transformers').SentenceTransformer('jinaai/jina-embeddings-v2-base-en', cache_folder="model_cache"),
        "intfloat_e5-base-v2": lambda: __import__('sentence_transformers').SentenceTransformer('intfloat/e5-base-v2'),
        "intfloat_e5-base": lambda: __import__('sentence_transformers').SentenceTransformer('intfloat/e5-base'),
        "nomic-ai_nomic-embed-text-v1.5": lambda: __import__('sentence_transformers').SentenceTransformer('nomic-ai/nomic-embed-text-v1.5'),
        #"Cohere_cohere-embed-english-v3.0": lambda: CohereEmbedder(model="embed-english-v3.0"),
        #"Cohere_cohere-embed-english-light-v3.0": lambda: CohereEmbedder(model="embed-english-light-v3.0"),
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
                # Create cache directory if it doesn't exist
                cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
                os.makedirs(cache_dir, exist_ok=True)
                
                # Try to load the model
                model = loader()
                if isinstance(model, NotImplementedError):
                    raise model
                
                # For sentence transformers models, ensure the model is downloaded
                if hasattr(model, 'download_model'):
                    model.download_model()
                    
            except Exception as e:
                # If the first attempt fails, try with direct model name
                try:
                    from sentence_transformers import SentenceTransformer
                    if '_' in model_name:
                        # Convert model name format
                        direct_name = model_name.replace('_', '/', 1)
                        logging.info(f"Attempting to load model with direct name: {direct_name}")
                        model = SentenceTransformer(direct_name, cache_folder=cache_dir)
                    else:
                        raise ValueError("Invalid model name format")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model '{model_name}'. First error: {e}. Second error: {e2}")
        else:
            # Fallback to appropriate model type based on name
            cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            if model_name.startswith('cross-encoder/'):
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(model_name, cache_folder=cache_dir)
            else:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name, cache_folder=cache_dir)
        
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
    import numpy as np
    model_name = emb_cfg.get('model', 'sentence-transformers_all_miniLM-L6-v2')
    batch_size = int(emb_cfg.get('batch_size', 64))
    parallelism = int(emb_cfg.get('parallelism', 1))
    normalize = emb_cfg.get('normalize', True)
    
    model, model_metrics = load_embedding_model(model_name)
    
    # Handle CrossEncoder models differently
    if isinstance(model, CrossEncoder):
        # For CrossEncoder, we'll extract embeddings from the base transformer model
        def get_embeddings(text_batch):
            # Ensure the model outputs hidden states
            model.model.config.output_hidden_states = True
            
            # Tokenize inputs
            inputs = model.tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Get model outputs
            with torch.no_grad():
                outputs = model.model(**inputs)
            
            # Get the hidden states from the last layer
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states[-1]
            else:
                # If no hidden states, use the last layer output
                hidden_states = outputs[0]
            
            # Apply attention mask and mean pooling
            attention_mask = inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            masked_embeddings = hidden_states * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return embeddings.cpu().numpy()

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = get_embeddings(batch)
            all_embeddings.append(batch_embeddings)
        
        # Stack all embeddings using numpy from the outer scope
        if not all_embeddings:
            raise ValueError("No embeddings were generated")
        embeddings = np.vstack(all_embeddings)
        
        # Apply normalization if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings, model_metrics
    
    # For regular sentence transformers, use encode method
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
