import torch
import numpy as np
import logging
from typing import List, Any, Dict, Optional, Tuple
from sklearn.decomposition import PCA
import os
import joblib
import sys

# Add the project root to Python path to allow importing from embedding_models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global variable to store PCA model
_pca_model = None

def load_pca_model(target_dim: int, pca_config: Optional[Dict] = None) -> Optional[PCA]:
    """Load the PCA model if it exists, otherwise return None.
    
    Args:
        target_dim: Target dimension for PCA reduction
        pca_config: Configuration dict from embed_config.dimension_reduction
    """
    global _pca_model
    if _pca_model is not None and _pca_model.n_components_ == target_dim:
        return _pca_model
    
    # Try to load from disk
    pca_file = f'pca_model_{target_dim}d.joblib'
    if os.path.exists(pca_file):
        model = joblib.load(pca_file)
        # Validate configuration matches
        if pca_config:
            if model.random_state != pca_config.get('random_state'):
                logging.warning("Loaded PCA model has different random_state than config")
            if model.whiten != pca_config.get('whiten', False):
                logging.warning("Loaded PCA model has different whiten setting than config")
        return model
    return None

def get_model_dimension(model_name: str) -> int:
    """Get the output dimension of the embedding model."""
    from embedding_models.embedder import load_embedding_model
    from sentence_transformers import SentenceTransformer, CrossEncoder
    
    # Load the model using our custom loader that handles different model types
    model, _ = load_embedding_model(model_name)
    
    # Handle different model types
    if isinstance(model, CrossEncoder):
        # For cross-encoders, use their base model's config
        hidden_size = model.model.config.hidden_size
        return hidden_size
    else:
        # For sentence transformers and other models that support encode
        dummy_embedding = model.encode(["dummy text"])
        if isinstance(dummy_embedding, torch.Tensor):
            dummy_embedding = dummy_embedding.cpu().numpy()
        return dummy_embedding.shape[1]

def generate_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
    normalize: bool,
    target_dim: int,
    use_pca: bool,
    pca_config: Optional[Dict]
) -> Any:
    """
    Generate embeddings for a list of texts using the specified model.
    
    Args:
        texts: List of input texts
        model_name: Name of the embedding model
        batch_size: Batch size for encoding
        normalize: Whether to normalize embeddings
        target_dim: Target dimension from config (required)
        use_pca: Whether to use PCA for dimension reduction
        pca_config: PCA configuration parameters
        
    Raises:
        ValueError: If model's output dimension doesn't match target_dim and PCA is disabled
    """
    # Check model's output dimension first
    model_dim = get_model_dimension(model_name)
    
    if model_dim != target_dim:
        if not use_pca:
            raise ValueError(
                f"Model dimension mismatch: {model_name} outputs {model_dim}-dimensional vectors, "
                f"but config requires {target_dim} dimensions. Either:\n"
                f"1. Set dimension={model_dim} in config.yaml to match the model's output, or\n"
                f"2. Enable PCA reduction by setting use_pca: true in config.yaml"
            )
        else:
            logging.info(f"[Dimension Reduction] Model {model_name} outputs {model_dim}-d vectors. "
                        f"Will use PCA to reduce to {target_dim}-d as configured.")
            if model_dim < target_dim:
                raise ValueError(
                    f"Cannot use PCA to increase dimensions from {model_dim} to {target_dim}. "
                    f"Target dimension must be smaller than model output dimension."
                )
    elif use_pca:
        logging.warning(f"PCA is enabled but not needed - model dimension ({model_dim}) "
                       f"already matches target dimension ({target_dim}). PCA will be skipped.")
    
    # Use embedder to get model and track resources
    from embedding_models.embedder import embed_texts
    
    # Generate embeddings and get metrics
    embeddings, model_metrics = embed_texts(texts, {
        'model': model_name,
        'batch_size': batch_size,
        'normalize': normalize,
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    })
    
    # Ensure embeddings are in numpy array format
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    
    # Always check dimensions and apply PCA if needed
    # Save PCA model for query transformation if needed
    global _pca_model
    current_dim = embeddings.shape[1]
    if current_dim != target_dim:
        if use_pca:
            n_samples, n_features = embeddings.shape
            n_components = min(target_dim, n_samples, n_features)
            if target_dim > n_components:
                logging.warning(f"[PCA] Requested n_components={target_dim} exceeds min(n_samples, n_features)={n_components}. Using n_components={n_components} instead.")
            logging.info(f"Reducing embedding dimension from {current_dim} to {n_components} using PCA")
            pca_params = pca_config or {}
            global _pca_model
            _pca_model = PCA(
                n_components=n_components,
                random_state=pca_params.get('random_state', 42),
                whiten=pca_params.get('whiten', False)
            )
            embeddings = _pca_model.fit_transform(embeddings)
            # Save PCA model for later use with queries
            pca_file = f'pca_model_{n_components}d.joblib'
            joblib.dump(_pca_model, pca_file)
            logging.info(f"Saved PCA model to {pca_file}")
            # Always normalize after PCA to ensure consistent magnitudes
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Update target_dim for final check
            target_dim = n_components
        else:
            raise ValueError(
                f"Dimension mismatch: Model {model_name} produced {current_dim}-d vectors, "
                f"but target dimension is {target_dim}. Enable PCA reduction or use matching dimensions."
            )
    # Validate final dimensions (allow for auto-adjusted target_dim)
    if embeddings.shape[1] != target_dim:
        raise ValueError(
            f"Final embedding dimension ({embeddings.shape[1]}) does not match target ({target_dim}). "
            "This should not happen - please report this error."
        )
    
    return embeddings, model_metrics
