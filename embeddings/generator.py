import torch
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Any, Dict, Optional
from sklearn.decomposition import PCA

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
    
    # Check if we need to apply PCA
    if use_pca and embeddings.shape[1] != target_dim:
        print(f"Reducing embedding dimension from {embeddings.shape[1]} to {target_dim} using PCA")
        pca_params = pca_config or {}
        pca = PCA(
            n_components=target_dim,
            random_state=pca_params.get('random_state', 42),
            whiten=pca_params.get('whiten', False)
        )
        embeddings = pca.fit_transform(embeddings)
        
        # Normalize after PCA if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings
