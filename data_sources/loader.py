import os
import pandas as pd
import pdfplumber
from sqlalchemy import create_engine
from typing import List, Dict, Tuple, Any
from chunking import create_chunker
import logging


def process_pdf(file_path: str, chunk_size: int = 1000) -> List[Dict]:
    """
    Process a PDF file and return a list of dictionaries containing page content and metadata.
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Maximum number of characters per chunk
        
    Returns:
        List of dictionaries with text content and metadata
    """
    documents = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    # Split text into chunks if it exceeds chunk_size
                    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                    for chunk_num, chunk in enumerate(text_chunks, 1):
                        doc = {
                            'content': chunk.strip(),
                            'page_number': page_num,
                            'chunk_number': chunk_num,
                            'total_chunks': len(text_chunks),
                            'filename': os.path.basename(file_path)
                        }
                        documents.append(doc)
    except Exception as e:
        raise ValueError(f"Error processing PDF file {file_path}: {str(e)}")
    
    return documents

def prepare_texts_for_embedding(df: pd.DataFrame, embed_columns: List[str], config: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Prepare texts for embedding while preserving original columns for payload.
    Applies chunking strategy if configured.
    
    Args:
        df: Input DataFrame
        embed_columns: Columns to include in embedding
        config: Configuration dictionary with chunking settings
    
    Returns:
        Tuple of (texts for embedding, payload dictionaries)
    """
    if embed_columns == 'all':
        embed_columns = df.columns.tolist()
    
    texts = []
    payloads = []
    
    # Create chunker from config
    chunker = create_chunker(config.get('chunking', {'strategy': 'none'}))
    
    for _, row in df.iterrows():
        # Create the concatenated text for embedding
        pairs = [f"{col}: {row[col]}" for col in embed_columns if col in df.columns]
        text = "; ".join(pairs)
        
        # Apply chunking strategy
        chunks = chunker.chunk_text(text)
        
        for chunk in chunks:
            texts.append(chunk['content'])
            
            # Create payload dictionary with original values and chunk metadata
            payload = {
                **{col: row[col] for col in embed_columns if col in df.columns},
                **{
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'chunking_strategy': chunk['strategy'],
                    'chunk_size': chunk['chunk_size']
                }
            }
            payloads.append(payload)
    
    logging.info(f"Generated {len(texts)} chunks using {chunker.strategy} strategy")
    return texts, payloads

def load_data(config):
    """Load data and return DataFrame, collection name, texts and payloads"""
    ds = config['data_source']
    
    # Use the collection name from vector_db config instead of file path
    collection_name = config['vector_db'].get('collection', 'flight_embeddings')
    
    def get_file_paths(file_pattern: str) -> List[str]:
        """Get list of files matching the pattern"""
        import glob
        if not os.path.isabs(file_pattern):
            # Make path absolute if it's relative
            file_pattern = os.path.join(os.getcwd(), file_pattern)
        files = glob.glob(file_pattern)
        if not files:
            raise ValueError(f"No files found matching pattern: {file_pattern}")
        logging.info(f"Found {len(files)} files matching pattern: {file_pattern}")
        return files
    
    if ds['type'] == 'csv':
        file_path = ds['file']
        if not os.path.isabs(file_path):
            file_path = os.path.join(
                os.path.dirname(__file__)
            )
        df = pd.read_csv(file_path)
        
        # Prepare texts and payloads
        embed_columns = ds.get('embed_columns', df.columns.tolist())
        texts, payloads = prepare_texts_for_embedding(df, embed_columns)
        
        return df, collection_name, texts, payloads
    
    elif ds['type'] == 'oracle':
        engine = create_engine(f"oracle+cx_oracle://{ds['user']}:{ds['password']}@{ds['host']}:{ds['port']}/{ds['sid']}")
        df = pd.read_sql(f"SELECT * FROM {ds['table']}", engine)
        collection_name = ds['table']
        # If embed_columns is specified for oracle
        embed_columns = ds.get('embed_columns', None)
        if embed_columns:
            texts, payloads = prepare_texts_for_embedding(df, embed_columns)
    
    elif ds['type'] in ['json', 'txt']:
        if ds['type'] == 'json':
            df = pd.read_json(ds['file'])
            embed_columns = ds.get('embed_columns', None)
            if embed_columns:
                texts, payloads = prepare_texts_for_embedding(df, embed_columns)
        elif ds['type'] == 'txt':
            df = pd.read_csv(ds['file'], delimiter=config.get('chunking', {}).get('delimiter', '\n'), header=None)
            embed_columns = ds.get('embed_columns', None)
            if embed_columns:
                texts, payloads = prepare_texts_for_embedding(df, embed_columns)
        collection_name = ds['file'].split('.')[0]
    elif ds['type'] == 'pdf':
        file_pattern = ds['file']
        files = get_file_paths(file_pattern)
        
        # Process all PDF files
        all_documents = []
        chunk_size = config.get('chunking', {}).get('chunk_size', 1000)
        
        for file_path in files:
            try:
                logging.info(f"Processing PDF file: {os.path.basename(file_path)}")
                documents = process_pdf(file_path, chunk_size)
                all_documents.extend(documents)
                logging.info(f"Extracted {len(documents)} chunks from {os.path.basename(file_path)}")
            except Exception as e:
                logging.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
                continue
        
        if not all_documents:
            raise ValueError("No valid documents were processed from any of the PDF files")
        
        logging.info(f"Total chunks extracted from all PDFs: {len(all_documents)}")
        
        # Convert all documents to DataFrame
        df = pd.DataFrame(all_documents)
        
        # Prepare texts and payloads
        texts = df['content'].tolist()
        payloads = df.to_dict('records')
        
        # Use filename without extension as collection name if not specified
        if not collection_name:
            collection_name = os.path.splitext(os.path.basename(file_path))[0]
            
        return df, collection_name, texts, payloads
    else:
        raise ValueError(f"Unsupported data source type: {ds['type']}")
    
    return df, collection_name, texts, payloads
