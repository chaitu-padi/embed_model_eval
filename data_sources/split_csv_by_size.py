import os
import pandas as pd
import math

def split_csv_by_size(input_path, output_dir, sizes_mb):
    """
    Splits a large CSV file into multiple files of specified sizes (in MB).
    Args:
        input_path (str): Path to the input CSV file.
        output_dir (str): Directory to save the output files.
        sizes_mb (list): List of sizes in MB for output files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Get total file size in bytes
    total_size = os.path.getsize(input_path)
    # Calculate number of rows per MB (approximate)
    sample = pd.read_csv(input_path, nrows=10000)
    sample_size = sample.memory_usage(deep=True).sum()
    rows_per_mb = int(10000 / (sample_size / (1024*1024)))
    # Read in chunks and write to files
    reader = pd.read_csv(input_path, chunksize=rows_per_mb)
    sizes_bytes = [size * 1024 * 1024 for size in sizes_mb]
    file_idx = 0
    written_bytes = 0
    out_file = None
    out_path = None
    for chunk in reader:
        if out_file is None:
            out_path = os.path.join(output_dir, f"split_{sizes_mb[file_idx]}MB.csv")
            out_file = open(out_path, "w", encoding="utf-8")
            chunk.to_csv(out_file, index=False, header=True)
            written_bytes = out_file.tell()
        else:
            chunk.to_csv(out_file, index=False, header=False)
            written_bytes = out_file.tell()
        if written_bytes >= sizes_bytes[file_idx]:
            out_file.close()
            file_idx += 1
            out_file = None
            if file_idx >= len(sizes_mb):
                break
    if out_file is not None:
        out_file.close()
    print(f"Created {file_idx} files in {output_dir}")

if __name__ == "__main__":
    sizes_mb = [2, 5] #,10, 500, 750, 1000]
    input_path="C:\\Users\\Lenovo\\PycharmProjects\\GenAILC\\embed_model_eval\\data\\Combined_Flights_2022.csv"
    output_path="C:\\Users\\Lenovo\\PycharmProjects\\GenAILC\\embed_model_eval\\data\\output"
    split_csv_by_size(input_path, output_path, sizes_mb)
