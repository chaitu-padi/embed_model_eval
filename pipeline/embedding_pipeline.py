import logging
import time
from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import multiprocessing

class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class EmbeddingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = min(multiprocessing.cpu_count(), 8)  # Use up to 8 CPU cores
        self.embeddings = []
        self.texts = []
        self.payloads = []
        self.metrics = {}

    def init_model(self) -> None:
        """Initialize the embedding model"""
        emb_cfg = self.config['embed_config']
        self.model = SentenceTransformer(
            emb_cfg.get('model', 'all-MiniLM-L6-v2'), 
            backend='onnx'
        )

    def generate_embeddings(self, texts: List[str]) -> Tuple[List[Any], float]:
        """Generate embeddings using parallel processing"""
        emb_cfg = self.config['embed_config']
        batch_size = int(emb_cfg.get('batch_size', 128))  # Increased batch size
        normalize = emb_cfg.get('normalize', True)
        
        # Create dataset and dataloader for parallel processing
        dataset = TextDataset(texts)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

        embeddings = []
        start_time = time.time()
        
        # Move model to GPU if available
        self.model = self.model.to(self.device)
        
        # Enable model half precision for faster computation
        if torch.cuda.is_available():
            self.model.half()

        # Enable evaluation mode for inference
        self.model.eval()
        
        with torch.no_grad():  # Disable gradient computation
            for batch in tqdm(dataloader, desc="[Embedding] Generating embeddings", unit="batch"):
                # Process batch
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    normalize_embeddings=normalize,
                    device=self.device
                )
                
                # Move to CPU and convert to numpy
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings)

                # Log progress
                logging.info(f"[Embedding] Progress: {len(embeddings)}/{len(texts)} texts embedded.")

        embedding_time = time.time() - start_time
        
        # Convert to numpy array for consistency
        embeddings = np.array(embeddings)
        
        return embeddings, embedding_time

    def insert_to_vector_db(self, embeddings: List[Any], texts: List[str], payloads: List[Dict]) -> Tuple[Any, float]:
        """Insert embeddings into vector database"""
        from vector_databases.insertion import insert_embeddings_qdrant, setup_qdrant_indexing
        
        vdb_cfg = self.config['vector_db']
        db_type = vdb_cfg.get('type', 'qdrant').lower()
        host = vdb_cfg.get('host', 'localhost')
        port = int(vdb_cfg.get('port', 6333))
        collection = vdb_cfg.get('collection', 'flight_embeddings')  # Use fixed collection name
        
        if db_type == 'qdrant':
            emb_cfg = self.config['embed_config']
            vector_size = emb_cfg.get('dimension', 768)
            batch_size = vdb_cfg.get('batch_size', 100)
            
            # Use the sanitized collection name throughout
            client, insertion_time = insert_embeddings_qdrant(
                embeddings=embeddings,
                texts=texts,
                payloads=payloads,
                collection_name=collection,  # Use the fixed collection name
                vector_size=vector_size,
                host=host,
                port=port,
                batch_size=batch_size
            )
            
            # Setup indexing with the same collection name
            vector_index_cfg = vdb_cfg.get('vector_index', {})
            payload_index_cfg = vdb_cfg.get('payload_index', [])
            setup_qdrant_indexing(client, collection, vector_index_cfg, payload_index_cfg)
            
            return client, insertion_time
        else:
            raise ValueError(f"Unsupported vector DB type: {db_type}")

    def setup_qdrant_indexing(self, client: Any, collection_name: str) -> None:
        """Configure Qdrant indexing with proper collection creation"""
        from qdrant_client.models import Distance, VectorParams
        
        vdb_cfg = self.config['vector_db']
        vector_index_cfg = vdb_cfg.get('vector_index', {})
        payload_index_cfg = vdb_cfg.get('payload_index', [])
        emb_cfg = self.config['embed_config']
        vector_size = emb_cfg.get('dimension', 768)

        try:
            # First check if collection exists
            collections = client.get_collections().collections
            collection_exists = any(col.name == collection_name for col in collections)

            if not collection_exists:
                # Create collection if it doesn't exist
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logging.info(f"Created new collection: {collection_name}")

            # Update collection settings
            if vector_index_cfg:
                index_type = vector_index_cfg.get('type', 'hnsw')
                index_params = vector_index_cfg.get('params', {})
                
                hnsw_config = None
                if index_type == 'hnsw':
                    hnsw_config = {
                        k: v for k, v in index_params.items() 
                        if k in ['m', 'ef_construct']
                    }
                
                client.update_collection(
                    collection_name=collection_name,
                    optimizer_config={"default_segment_number": 2},
                    hnsw_config=hnsw_config
                )
                logging.info(f"Updated collection indexing settings: {collection_name}")

            # Create payload indices
            for field_cfg in payload_index_cfg:
                field = field_cfg.get('field')
                idx_type = field_cfg.get('type')
                if field and idx_type:
                    try:
                        client.create_payload_index(
                            collection_name=collection_name,
                            field_name=field,
                            field_schema=idx_type
                        )
                        logging.info(f"Created payload index for field: {field}")
                    except Exception as e:
                        logging.warning(f"Failed to create payload index for {field}: {str(e)}")

        except Exception as e:
            logging.error(f"Error setting up Qdrant indexing: {str(e)}")
            raise