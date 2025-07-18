from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

def embed_texts(texts, emb_cfg):
    model_name = emb_cfg.get('model', 'all-MiniLM-L6-v2')
    batch_size = int(emb_cfg.get('batch_size', 64))
    parallelism = int(emb_cfg.get('parallelism', 1))
    normalize = emb_cfg.get('normalize', True)
    model = SentenceTransformer(model_name)  # Load once!

    def embed_batch(batch):
        return model.encode(batch, show_progress_bar=False, batch_size=batch_size, normalize_embeddings=normalize)

    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    embeddings = []
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        results = list(executor.map(embed_batch, batches))
    for batch_emb in results:
        embeddings.extend(batch_emb)
    return embeddings
