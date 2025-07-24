import torch
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Any, Dict, Optional
from sklearn.decomposition import PCA
import os
import joblib

# Global variable to store PCA model
_pca_model = None

def load_pca_model(target_dim: int, pca_config: Optional[Dict] = None) -> Optional[PCA]:
    """Load the PCA model if it exists, otherwise return None."""
    global _pca_model
    if _pca_model is not None and _pca_model.n_components_ == target_dim:
        return _pca_model
    
    # Try to load from disk
    pca_file = f'pca_model_{target_dim}d.joblib'
    if os.path.exists(pca_file):
        return joblib.load(pca_file)
    return None

def get_model_dimension(model_name: str) -> int:
    """Get the output dimension of a SentenceTransformer model."""
    temp_model = SentenceTransformer(model_name)
    # Use a dummy input to get the output dimension
    dummy_embedding = temp_model.encode(["dummy text"], convert_to_numpy=True)
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
    Generate embeddings for a list of texts using SentenceTransformer.
    
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    # Generate initial embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        device=device
    )
    
    # Always check dimensions and apply PCA if needed
    # Save PCA model for query transformation if needed
    global _pca_model
    current_dim = embeddings.shape[1]
    if current_dim != target_dim:
        if use_pca:
            logging.info(f"Reducing embedding dimension from {current_dim} to {target_dim} using PCA")
            pca_params = pca_config or {}
            # Create and fit PCA model
            global _pca_model
            _pca_model = PCA(
                n_components=target_dim,
                random_state=pca_params.get('random_state', 42),
                whiten=pca_params.get('whiten', False)
            )
            embeddings = _pca_model.fit_transform(embeddings)
            
            # Save PCA model for later use with queries
            pca_file = f'pca_model_{target_dim}d.joblib'
            joblib.dump(_pca_model, pca_file)
            logging.info(f"Saved PCA model to {pca_file}")
            
            # Always normalize after PCA to ensure consistent magnitudes
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        else:
            raise ValueError(
                f"Dimension mismatch: Model {model_name} produced {current_dim}-d vectors, "
                f"but target dimension is {target_dim}. Enable PCA reduction or use matching dimensions."
            )
    
    # Validate final dimensions
    if embeddings.shape[1] != target_dim:
        raise ValueError(
            f"Final embedding dimension ({embeddings.shape[1]}) does not match target ({target_dim}). "
            "This should not happen - please report this error."
        )
    
    return embeddings
