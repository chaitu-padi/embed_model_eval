def print_report(model_names, ds, db_type, host, port, collection, all_metrics, top_k, dimension=None, batch_size=None, parallelism=None, report_filename=None, config=None):
    """
    Generate a comparative HTML report for multiple embedding models.
    
    Args:
        model_names (list): List of model names to compare
        model_names (list): List of model names to compare): Data source configuration
        db_type (str): Database type
        host (str): Database host
        port (str/int): Database port
        collection (str): Collection name
        all_metrics (dict): Dictionary of metrics for each model
        top_k (int): Number of top results for retrieval
        dimension (int, optional): Embedding dimension
        batch_size (int, optional): Batch size used
        parallelism (int, optional): Parallelism level used
        report_filename (str, optional): Custom report filename
        config (dict, optional): Full configuration dictionary
    """
    import torch
    import datetime
    import os
    import platform
    import psutil
    
    # Set default config if not provided
    if config is None:
        config = {}
        
    # Generate default report filename if not provided
    if report_filename is None:
        now = datetime.datetime.now()
        date_str = now.strftime('%Y-%m-%d_%H-%M-%S')
        report_filename = f"model_evaluation_report_{date_str}"
    
    # Gather additional details
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    data_path = ds.get('file', ds.get('table', ''))
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 'N/A'
    now = datetime.datetime.now()
    date_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    
    # Generate HTML report filename
    report_filename = f"{report_filename}.html"
    
    # HTML Report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Embedding Model Evaluation Report</title>
        <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f8f8;
        }}
        .report {{
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 32px;
            margin: 32px auto;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            text-align: center;
            margin-top: 0;
        }}
        h3 {{
            color: #2980b9;
            margin-top: 30px;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background-color: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: normal;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .module-header {{
            background-color: #2c3e50;
            color: white;
            font-weight: bold;
        }}
        .metric-value {{
            font-family: 'Courier New', Courier, monospace;
            color: #2c3e50;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
            text-align: center;
        }}
        .nav-section {{
            margin: 20px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .nav-section ul {{
            list-style-type: none;
            padding: 0;
            margin: 0;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .nav-section li {{
            margin: 5px;
            padding: 5px 10px;
            background-color: #e9ecef;
            border-radius: 3px;
        }}
        .nav-section a {{
            text-decoration: none;
            color: #2c3e50;
        }}
        .nav-section a:hover {{
            color: #3498db;
        }}
        .performance {{
            background: #e8f5e9;
        }}
        </style>
    </head>
    <body>
    <div class="report">
        <h2>Embedding Model Evaluation Report</h2>
        <div class="timestamp">Generated: {now}</div>
        
        <div class="nav-section">
            <ul>
                <li><a href="#hardware">1. Hardware Configuration</a></li>
                <li><a href="#dataset">2. Dataset Configuration</a></li>
                <li><a href="#embedding">3. Embedding Model Configuration</a></li>
                <li><a href="#vectordb">4. Vector Database Configuration</a></li>
                <li><a href="#retrieval">5. Retrieval Configuration</a></li>
                <li><a href="#performance">6. Performance Metrics</a></li>
            </ul>
        </div>

    <!-- 1. Hardware Configuration -->
    <h3 id="hardware">1. Hardware Configuration</h3>
    <table>
        <tr>
            <th style="width: 30%;">Parameter</th>
            <th>Value</th>
        </tr>
        <tr class="module-header">
            <td colspan="2">System Specifications</td>
        </tr>
        <tr>
            <td>CPU Model</td>
            <td class="metric-value">{platform.processor()}</td>
        </tr>
        <tr>
            <td>CPU Cores</td>
            <td class="metric-value">{os.cpu_count()}</td>
        </tr>
        <tr>
            <td>Total RAM (GB)</td>
            <td class="metric-value">{round(psutil.virtual_memory().total / (1024**3), 2)}</td>
        </tr>
        <tr class="module-header">
            <td colspan="2">GPU Information</td>
        </tr>
        <tr>
            <td>GPU Device</td>
            <td class="metric-value">{device}</td>
        </tr>
        <tr>
            <td>GPU Model</td>
            <td class="metric-value">{device_name}</td>
        </tr>
        <tr>
            <td>Device Type</td>
            <td class="metric-value">{device}</td>
        </tr>
        <tr>
            <td>Device Name</td>
            <td class="metric-value">{device_name}</td>
        </tr>
        <tr>
            <td>CUDA Available</td>
            <td class="metric-value">{"Yes" if torch.cuda.is_available() else "No"}</td>
        </tr>
    </table>

    <!-- 2. Dataset Configuration -->
    <h3 id="dataset">2. Dataset Configuration</h3>
    <table>
        <tr>
            <th style="width: 30%;">Parameter</th>
            <th>Value</th>
        </tr>
        <tr class="module-header">
            <td colspan="2">Data Source Information</td>
        </tr>
        <tr>
            <td>Dataset Source</td>
            <td class="metric-value">{os.path.basename(data_path)}</td>
        </tr>
        <tr>
            <td>Dataset Size</td>
            <td class="metric-value">{data_size if isinstance(data_size, str) else f"{data_size / (1024*1024):.2f} MB"}</td>
        </tr>
        <tr>
            <td>Source Type</td>
            <td class="metric-value">{ds.get('type', 'N/A')}</td>
        </tr>
        <tr>
            <td>Total Records</td>
            <td class="metric-value">{ds.get('num_records', 'N/A')}</td>
        </tr>
        <tr class="module-header">
            <td colspan="2">Data Processing</td>
        </tr>
        <tr>
            <td>Chunk Strategy</td>
            <td class="metric-value">{config.get('chunking', {}).get('strategy', 'N/A')}</td>
        </tr>
        <tr>
            <td>Chunk Size</td>
            <td class="metric-value">{config.get('chunking', {}).get('chunk_size', 'N/A')}</td>
        </tr>
        <tr>
            <td>Overlap</td>
            <td class="metric-value">{config.get('chunking', {}).get('overlap', 'N/A')}</td>
        </tr>
    </table>

    <!-- 3. Embedding Model Comparison -->
    <h3 id="embedding">3. Model Configuration and Performance Comparison</h3>
    <table>
        <tr>
            <th style="width: 30%;">Parameter</th>
            {' '.join([f'<th>{model}</th>' for model in model_names])}
        </tr>
        <tr class="module-header">
            <td colspan="{len(model_names) + 1}">Model Information</td>
        </tr>
        <tr>
            <td>Model Type</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("model_type", "Dense Embedding")}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Output Dimension</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("dimension", dimension)}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Use PCA</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("use_pca", "No")}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>PCA Components</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("pca_components", "N/A")}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Normalize Embeddings</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("normalize", "No")}</td>' for model in model_names])}
        </tr>
        <tr class="module-header">
            <td colspan="{len(model_names) + 1}">Processing Parameters</td>
        </tr>
        <tr>
            <td>Embedding Batch Size</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("embedding_batch_size", "N/A")}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Vector DB Batch Size</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("db_batch_size", "N/A")}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Parallelism</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("parallelism", parallelism)}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Device</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("device", device)}</td>' for model in model_names])}
        </tr>
        <tr class="module-header">
            <td colspan="{len(model_names) + 1}">Resource Usage</td>
        </tr>
        <tr>
            <td>Memory Usage (MB)</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("model_memory_mb", "N/A")}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Loading Time (s)</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("model_load_time", "N/A")}</td>' for model in model_names])}
        </tr>
    </table>

    <!-- 4. Vector Database Configuration -->
    <h3 id="vectordb">4. Vector Database Configuration</h3>
    <table>
        <tr>
            <th style="width: 30%;">Parameter</th>
            <th>Value</th>
        </tr>
        <tr class="module-header">
            <td colspan="2">Database Connection</td>
        </tr>
        <tr>
            <td>Database Type</td>
            <td class="metric-value">{db_type}</td>
        </tr>
        <tr>
            <td>Host:Port</td>
            <td class="metric-value">{host}:{port}</td>
        </tr>
        <tr>
            <td>Collection</td>
            <td class="metric-value">{collection}</td>
        </tr>
        <tr class="module-header">
            <td colspan="2">Insertion Configuration</td>
        </tr>
        <tr>
            <td>Insertion Batch Size</td>
            <td class="metric-value">{all_metrics[model_names[0]]['db_batch_size']}</td>
        </tr>
        <tr>
            <td>Upsert Retries</td>
            <td class="metric-value">{all_metrics[model_names[0]]['db_upsert_retries']}</td>
        </tr>
        <tr>
            <td>Retry Delay (s)</td>
            <td class="metric-value">{all_metrics[model_names[0]]['db_retry_delay']}</td>
        </tr>
        <tr>
            <td>Vector Distance</td>
            <td class="metric-value">{all_metrics[model_names[0]].get('distance_type', 'COSINE')}</td>
        </tr>
    </table>
    <!-- 5. Vector Database Retrieval Configuration -->
    <h3>5. Vector Database Retrieval Configuration</h3>
    <table>
        <tr>
            <th style="width: 30%;">Parameter</th>
            <th>Value</th>
        </tr>
        <tr class="module-header">
            <td colspan="2">Search Configuration</td>
        </tr>
        <tr>
            <td>Search Type</td>
            <td class="metric-value">{all_metrics[model_names[0]].get('search_type', 'semantic')}</td>
        </tr>
        <tr>
            <td>Top-K Results</td>
            <td class="metric-value">{all_metrics[model_names[0]].get('top_k', top_k)}</td>
        </tr>
        <tr>
            <td>Distance Metric</td>
            <td class="metric-value">{all_metrics[model_names[0]].get('distance_metric', 'COSINE')}</td>
        </tr>
        <tr class="module-header">
            <td colspan="2">Query Configuration</td>
        </tr>
        <tr>
            <td>Semantic Query</td>
            <td class="metric-value">{all_metrics[model_names[0]].get('semantic_query', 'N/A')}</td>
        </tr>
        <tr>
            <td>Filter Applied</td>
            <td class="metric-value">{str(all_metrics[model_names[0]].get('filter_applied', 'No'))}</td>
        </tr>
    </table>

    <!-- 6. Performance Metrics -->
    <h3 id="performance">6. Performance Metrics</h3>
    <table>
        <tr>
            <th style="width: 30%;">Metric</th>
            {' '.join([f'<th>{model}</th>' for model in model_names])}
        </tr>
        <tr class="module-header">
            <td colspan="{len(model_names) + 1}">Pipeline Timing Analysis</td>
        </tr>
        <tr>
            <td>Data Loading Phase</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("data_load_time", "N/A"):.2f} seconds</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Embedding Generation Phase</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("embedding_time", "N/A"):.2f} seconds</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Database Insertion Phase</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("insertion_time", "N/A"):.2f} seconds</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Semantic Retrieval Phase</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("retrieval_time", "N/A"):.2f} seconds</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Total Pipeline Time</td>
            {' '.join([f'<td class="metric-value">{sum([all_metrics[model].get(k, 0) for k in ["data_load_time", "embedding_time", "insertion_time", "retrieval_time"]]):.2f} seconds</td>' for model in model_names])}
        </tr>
        <tr class="module-header">
            <td colspan="{len(model_names) + 1}">Performance Metrics</td>
        </tr>
        <tr>
            <td>Processing Rate</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("processing_rate", "N/A")} records/second</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Embedding Generation Rate</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("embeddings_per_second", "N/A")} embeddings/second</td>' for model in model_names])}
        </tr>
        <tr class="module-header">
            <td colspan="{len(model_names) + 1}">Resource Usage</td>
        </tr>
        <tr>
            <td>Memory Peak</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("memory_peak_mb", "N/A")} MB</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Total Embeddings</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("total_embeddings", "N/A")}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>GPU Memory Used</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("gpu_memory_used", "N/A")} GB</td>' for model in model_names])}
        </tr>
        <tr>
            <td>GPU Utilization</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("gpu_utilization", "N/A")}%</td>' for model in model_names])}
        </tr>
        <tr class="module-header">
            <td colspan="{len(model_names) + 1}">Vector Database Metrics</td>
        </tr>
        <tr>
            <td>Index Build Time</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("index_build_time", "N/A")} seconds</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Insert Rate</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("insert_rate", "N/A")} vectors/second</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Average Query Time</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("avg_query_time", "N/A")} seconds</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Query Throughput</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("query_throughput", "N/A")} queries/second</td>' for model in model_names])}
        </tr>
        <tr class="module-header">
            <td colspan="{len(model_names) + 1}">Quality Metrics</td>
        </tr>
        <tr>
            <td>Accuracy</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("accuracy", "N/A"):.3f}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Recall@{top_k}</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("recall", "N/A"):.3f}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Precision@{top_k}</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("precision", "N/A"):.3f}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>F1 Score</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("f1", "N/A"):.3f}</td>' for model in model_names])}
        </tr>
        <tr>
            <td>Mean Reciprocal Rank</td>
            {' '.join([f'<td class="metric-value">{all_metrics[model].get("mrr", "N/A")}</td>' for model in model_names])}
        </tr>
    </table>
    
    <!-- Execution Summary -->
    <h3>Execution Summary</h3>
    <table>
        <tr>
            <th style="width: 30%;">Category</th>
            <th>Performance Summary</th>
        </tr>
        <tr class="module-header">
            <td colspan="2">Resource Utilization</td>
        </tr>
        <tr>
            <td>CPU Usage</td>
            <td class="metric-value">Peak: {all_metrics[model_names[0]].get('peak_cpu_usage', 'N/A')}%, Average: {all_metrics[model_names[0]].get('avg_cpu_usage', 'N/A')}%</td>
        </tr>
        <tr>
            <td>Memory Usage</td>
            <td class="metric-value">Peak: {all_metrics[model_names[0]].get('peak_memory_mb', 'N/A')} MB, Average: {all_metrics[model_names[0]].get('avg_memory_mb', 'N/A')} MB</td>
        </tr>
        <tr>
            <td>GPU Resources</td>
            <td class="metric-value">Memory Used: {all_metrics[model_names[0]].get('gpu_memory_used', 'N/A')} GB, Utilization: {all_metrics[model_names[0]].get('gpu_utilization', 'N/A')}%</td>
        </tr>
        <tr class="module-header">
            <td colspan="2">Processing Pipeline</td>
        </tr>
        <tr>
            <td>Data Processing</td>
            <td class="metric-value">Processed {all_metrics[model_names[0]].get('total_records', 'N/A')} records at {all_metrics[model_names[0]].get('processing_rate', 'N/A')} records/second</td>
        </tr>
        <tr>
            <td>Embedding Generation</td>
            <td class="metric-value">Generated {all_metrics[model_names[0]].get('total_embeddings', 'N/A')} embeddings at {all_metrics[model_names[0]].get('embeddings_per_second', 'N/A')} embeddings/second (Batch Size: {all_metrics[model_names[0]].get('embedding_batch_size', 'N/A')})</td>
        </tr>
        <tr>
            <td>Database Operations</td>
            <td class="metric-value">Indexed and inserted {all_metrics[model_names[0]].get('total_embeddings', 'N/A')} vectors into {db_type} collection '{collection}' (Batch Size: {all_metrics[model_names[0]].get('db_batch_size', 'N/A')})</td>
        </tr>
        <tr class="module-header">
            <td colspan="2">Overall Performance</td>
        </tr>
        <tr>
            <td>Quality Metrics</td>
            <td class="metric-value">Achieved Precision@{top_k}: {all_metrics[model_names[0]].get('precision', 0.0):.3f}, Recall@{top_k}: {all_metrics[model_names[0]].get('recall', 0.0):.3f}, F1: {all_metrics[model_names[0]].get('f1', 0.0):.3f}</td>
        </tr>
        <tr>
            <td>Time Distribution</td>
            <td class="metric-value">Processing: {all_metrics[model_names[0]].get('data_load_time', 'N/A')}s, Embedding: {all_metrics[model_names[0]].get('embedding_time', 'N/A')}s, Database: {all_metrics[model_names[0]].get('insertion_time', 'N/A')}s</td>
        </tr>
        <tr>
            <td>Final Assessment</td>
            <td class="metric-value">Successfully processed and indexed dataset with {all_metrics[model_names[0]].get('total_embeddings', 'N/A')} embeddings, achieving {all_metrics[model_names[0]].get('f1', 0.0):.3f} F1-score</td>
        </tr>
    </table>
    </div>
    </body>
    </html>
    """
    # Save HTML report with the dynamic filename
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(html)
    print("\n==== Embedding Model Performance Report ====")
    print(f"Report saved to {report_filename}")
    print("Summary:")
    for model_name in model_names:
        model_metrics = all_metrics[model_name]
        print(f"\nModel: {model_name}")
        print(f"Data Source: {ds['type']} ({data_path}) Size: {data_size} bytes")
        print(f"Vector DB: {db_type} ({host}:{port}) Collection: {collection}")
        print(f"PyTorch Device: {device} ({device_name})")
        print(f"Embedding Generation Time: {model_metrics.get('embedding_time', 0.0):.2f} seconds")
        print(f"Total Embeddings: {model_metrics.get('total_embeddings', 'N/A')}")
        print(f"Embedding Insertion Time: {model_metrics.get('insertion_time', 0.0):.2f} seconds")
        print(f"Embedding Batch Size: {model_metrics.get('embedding_batch_size', 'N/A')}")
        print(f"DB Batch Size: {model_metrics.get('db_batch_size', 'N/A')}")
        print(f"DB Retries: {model_metrics.get('db_upsert_retries', 'N/A')}")
        print(f"Retrieval Time: {model_metrics.get('retrieval_time', 0.0):.2f} seconds")
        print(f"Top-K: {model_metrics.get('top_k', top_k)}")
        print(f"Precision: {model_metrics.get('precision', 0.0):.3f}")
        print(f"Recall: {model_metrics.get('recall', 0.0):.3f}")
        print(f"F1 Score: {model_metrics.get('f1', 0.0):.3f}")
