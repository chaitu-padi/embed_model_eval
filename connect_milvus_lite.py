
# Milvus imports
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, CollectionStatus

def connect_milvus(collection_name: str, dim: int = 768, host: str = "milvus-standalone", port: str = "19530"):
    """
    Connects to Milvus Standalone (Docker) and ensures the collection exists.
    Returns the collection object.
    """
    try:
        connections.connect(uri=f"tcp://{host}:{port}")
        print(f"Connected to Milvus at tcp://{host}:{port}")
    except Exception as e:
        print(f"Failed to connect to Milvus at tcp://{host}:{port}. Error: {e}")
        raise

    # Check if collection exists
    if collection_name in [c.name for c in Collection.list()]:
        return Collection(collection_name)

    # Define schema
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Embedding collection")
    collection = Collection(collection_name, schema)
    return collection

def connect_qdrant(collection_name: str, dim: int = 768, host: str = "localhost", port: int = 6333):
    """
    Connects to Qdrant and ensures the collection exists.
    Returns the QdrantClient object.
    """
    client = QdrantClient(host=host, port=port)
    try:
        status = client.get_collection(collection_name=collection_name)
        print(f"Qdrant collection '{collection_name}' already exists and is healthy.")
    except Exception:
        print(f"Creating Qdrant collection '{collection_name}' with dim={dim}")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
    return client

if __name__ == "__main__":
    # Example usage Milvus
    #col = connect_milvus("demo_collection", dim=768)
    #print(f"Connected to Milvus Standalone. Collection '{col.name}' is ready.")

    # Example usage Qdrant
    qdrant_client = connect_qdrant("demo_collection", dim=768)
    print(f"Connected to Qdrant. Collection 'demo_collection' is ready.")
