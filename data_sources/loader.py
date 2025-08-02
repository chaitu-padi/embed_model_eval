import os
import pandas as pd
import pdfplumber
from sqlalchemy import create_engine
from typing import List, Dict, Tuple, Any
from chunking import create_chunker
import logging

# Suppress pdfminer/pdfplumber warnings
import logging as py_logging
for noisy_logger in ["pdfminer", "pdfminer.pdfinterp", "pdfplumber", "pdfinterp"]:
    py_logging.getLogger(noisy_logger).setLevel(py_logging.ERROR)


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
    logging.info(f"Processing PDF file: {file_path}")

    # First try with pdfplumber
    try:
        with pdfplumber.open(file_path) as pdf:
            logging.info(f"Successfully opened PDF with {len(pdf.pages)} pages")
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    logging.info(f"Processing page {page_num}")
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
                except Exception as page_error:
                    logging.warning(f"Error processing page {page_num}: {str(page_error)}. Trying alternative method...")
                    # Try alternative method for this page using PyPDF2
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                            if 0 <= page_num - 1 < len(pdf_reader.pages):
                                text = pdf_reader.pages[page_num - 1].extract_text()
                                if text:
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
                    except Exception as pypdf_error:
                        logging.error(f"Both PDF processing methods failed for page {page_num}: {str(pypdf_error)}")

    except Exception as e:
        logging.error(f"Error with primary PDF processing method: {str(e)}")
        # Try alternative method using PyPDF2
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                logging.info("Trying alternative PDF processing method with PyPDF2")
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text:
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
        except Exception as final_error:
            raise ValueError(f"All PDF processing methods failed for {file_path}. Original error: {str(e)}, Final error: {str(final_error)}")

    if not documents:
        logging.warning(f"No text content extracted from {file_path}")
        
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
    """Load data and return DataFrame, collection name, texts, payloads"""
    ds = config['data_source']
    
    # Use the collection name from vector_db config instead of file path
    collection_name = config['vector_db'].get('collection', 'flight_embeddings')
    
    # Initialize dataset statistics
    dataset_stats = {
        'total_size_bytes': 0,
        'num_files': 0,
        'num_chunks': 0,
        'avg_chunk_size': 0,
        'file_paths': []  # Store file paths for dataset size calculation
    }
    
    def get_file_paths(file_pattern: str) -> Tuple[List[str], int]:
        """Get list of files matching the pattern and total size in bytes"""
        import glob
        if not os.path.isabs(file_pattern):
            # Make path absolute if it's relative
            file_pattern = os.path.join(os.getcwd(), file_pattern)
        files = glob.glob(file_pattern)
        if not files:
            raise ValueError(f"No files found matching pattern: {file_pattern}")
            
        # Calculate total size
        total_size = sum(os.path.getsize(f) for f in files)
        
        # Store file paths for later use
        dataset_stats['file_paths'].extend(files)
        dataset_stats['total_size_bytes'] = total_size
        
        # Log detailed dataset information
        logging.info("Dataset Statistics:")
        logging.info(f"Number of files: {len(files)}")
        logging.info(f"Total dataset size: {total_size / (1024*1024):.2f} MB")
        logging.info(f"Average file size: {(total_size / len(files)) / (1024*1024):.2f} MB")
        
        # Get process memory usage
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logging.info(f"Current memory usage: {memory_info.rss / (1024*1024):.2f} MB")
        
        return files, total_size
    
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
        files, total_size = get_file_paths(file_pattern)
        
        # Update dataset statistics
        dataset_stats['total_size_bytes'] = total_size
        dataset_stats['num_files'] = len(files)
        
        # Process all PDF files
        all_documents = []
        total_content_size = 0
        chunk_size = config.get('chunking', {}).get('chunk_size', 1000)
        
        for file_path in files:
            try:
                logging.info(f"Processing PDF file: {os.path.basename(file_path)}")
                documents = process_pdf(file_path, chunk_size)
                all_documents.extend(documents)
                total_content_size += sum(len(doc['content']) for doc in documents)
                logging.info(f"Extracted {len(documents)} chunks from {os.path.basename(file_path)}")
            except Exception as e:
                logging.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
                continue
        
        if not all_documents:
            raise ValueError("No valid documents were processed from any of the PDF files")
        
        # Update dataset statistics
        dataset_stats['num_chunks'] = len(all_documents)
        dataset_stats['avg_chunk_size'] = total_content_size / len(all_documents) if all_documents else 0
        dataset_stats['total_content_size'] = total_content_size
        
        logging.info(f"Dataset Statistics:")
        logging.info(f"Total file size: {total_size / (1024*1024):.2f} MB")
        logging.info(f"Number of files: {len(files)}")
        logging.info(f"Number of chunks: {len(all_documents)}")
        logging.info(f"Average chunk size: {dataset_stats['avg_chunk_size']:.2f} characters")
        logging.info(f"Total extracted content size: {total_content_size / (1024*1024):.2f} MB")
        
        # Convert all documents to DataFrame
        df = pd.DataFrame(all_documents)
        
        # Prepare texts and payloads
        texts = df['content'].tolist()
        payloads = [
            {**doc, 'dataset_stats': dataset_stats}
            for doc in df.to_dict('records')
        ]
        
        # Use filename without extension as collection name if not specified
        if not collection_name:
            collection_name = os.path.splitext(os.path.basename(file_path))[0]
            
        return df, collection_name, texts, payloads
    else:
        raise ValueError(f"Unsupported data source type: {ds['type']}")
    
    return df, collection_name, texts, payloads
