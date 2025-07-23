import torch
import datetime
import os
import platform
import psutil

def print_report(model_names, ds, db_type, host, port, collection, all_metrics, top_k, dimension=None, batch_size=None, parallelism=None, report_filename=None, config=None):
    """
    Generate a comparative HTML report for multiple embedding models.
    
    Args:
        model_names (list): List of model names to compare
        ds (dict): Data source configuration
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
    # Set default config if not provided
    if config is None:
        config = {}
        
    # Generate default report filename if not provided
    if report_filename is None:
        now = datetime.datetime.now()
        date_str = now.strftime('%Y-%m-%d_%H-%M-%S')
        report_filename = f"model_evaluation_report_{date_str}"
    
    # Gather system details
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    data_path = ds.get('file', ds.get('table', ''))
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 'N/A'
    
    # Generate HTML header for models
    model_headers = ''.join([f'<th>{model}</th>' for model in model_names])
    
    def get_metric_rows(label, metric_key, format_func=lambda x: x):
        """Helper function to generate table rows for each metric"""
        values = ' '.join([
            f'<td class="metric-value">{format_func(all_metrics[model].get(metric_key, "N/A"))}</td>'
            for model in model_names
        ])
        return f'<tr><td>{label}</td>{values}</tr>'
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
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
        .best-value {{
            font-weight: bold;
            color: #27ae60;
        }}
        </style>
    </head>
    <body>
    <div class="report">
        <h2>Embedding Model Comparison Report</h2>
        <div class="timestamp">Generated: {datetime.datetime.now()}</div>

        <!-- System Configuration -->
        <h3>1. System Configuration</h3>
        <table>
            <tr>
                <th style="width: 20%;">Parameter</th>
                <th colspan="{len(model_names)}">Value</th>
            </tr>
            <tr>
                <td>CPU Model</td>
                <td colspan="{len(model_names)}">{platform.processor()}</td>
            </tr>
            <tr>
                <td>CPU Cores</td>
                <td colspan="{len(model_names)}">{os.cpu_count()}</td>
            </tr>
            <tr>
                <td>Total RAM</td>
                <td colspan="{len(model_names)}">{round(psutil.virtual_memory().total / (1024**3), 2)} GB</td>
            </tr>
            <tr>
                <td>GPU Device</td>
                <td colspan="{len(model_names)}">{device_name}</td>
            </tr>
            <tr>
                <td>CUDA Available</td>
                <td colspan="{len(model_names)}">{"Yes" if torch.cuda.is_available() else "No"}</td>
            </tr>
        </table>

        <!-- Dataset Information -->
        <h3>2. Dataset Configuration</h3>
        <table>
            <tr>
                <th style="width: 20%;">Parameter</th>
                <th colspan="{len(model_names)}">Value</th>
            </tr>
            <tr>
                <td>Dataset Source</td>
                <td colspan="{len(model_names)}">{os.path.basename(data_path)}</td>
            </tr>
            <tr>
                <td>Dataset Size</td>
                <td colspan="{len(model_names)}">{data_size if isinstance(data_size, str) else f"{data_size / (1024*1024):.2f} MB"}</td>
            </tr>
            <tr>
                <td>Source Type</td>
                <td colspan="{len(model_names)}">{ds.get('type', 'N/A')}</td>
            </tr>
            <tr>
                <td>Chunk Strategy</td>
                <td colspan="{len(model_names)}">{config.get('chunking', {}).get('strategy', 'N/A')}</td>
            </tr>
            <tr>
                <td>Chunk Size</td>
                <td colspan="{len(model_names)}">{config.get('chunking', {}).get('chunk_size', 'N/A')}</td>
            </tr>
        </table>

        <!-- Performance Comparison -->
        <h3>3. Performance Metrics</h3>
        <table>
            <tr>
                <th style="width: 20%;">Metric</th>
                {model_headers}
            </tr>
            <tr class="module-header">
                <td colspan="{len(model_names) + 1}">Timing Analysis (seconds)</td>
            </tr>
            {get_metric_rows("Data Load Time", "data_load_time", lambda x: f"{float(x):.2f}" if x != "N/A" else x)}
            {get_metric_rows("Embedding Time", "embedding_time", lambda x: f"{float(x):.2f}" if x != "N/A" else x)}
            {get_metric_rows("Insertion Time", "insertion_time", lambda x: f"{float(x):.2f}" if x != "N/A" else x)}
            {get_metric_rows("Retrieval Time", "retrieval_time", lambda x: f"{float(x):.2f}" if x != "N/A" else x)}
            
            <tr class="module-header">
                <td colspan="{len(model_names) + 1}">Processing Metrics</td>
            </tr>
            {get_metric_rows("Embeddings/Second", "embeddings_per_second", lambda x: f"{float(x):.2f}" if x != "N/A" else x)}
            {get_metric_rows("Total Embeddings", "total_embeddings")}
            {get_metric_rows("Processing Rate", "processing_rate")}

            <tr class="module-header">
                <td colspan="{len(model_names) + 1}">Quality Metrics</td>
            </tr>
            {get_metric_rows("Accuracy", "accuracy", lambda x: f"{float(x):.3f}" if x != "N/A" else x)}
            {get_metric_rows(f"Precision@{top_k}", "precision", lambda x: f"{float(x):.3f}" if x != "N/A" else x)}
            {get_metric_rows(f"Recall@{top_k}", "recall", lambda x: f"{float(x):.3f}" if x != "N/A" else x)}
            {get_metric_rows("F1 Score", "f1", lambda x: f"{float(x):.3f}" if x != "N/A" else x)}
        </table>

        <!-- Resource Usage -->
        <h3>4. Resource Usage</h3>
        <table>
            <tr>
                <th style="width: 20%;">Metric</th>
                {model_headers}
            </tr>
            {get_metric_rows("Peak CPU Usage", "peak_cpu_usage", lambda x: f"{float(x):.1f}%" if x != "N/A" else x)}
            {get_metric_rows("Peak Memory", "peak_memory_mb", lambda x: f"{float(x):.1f} MB" if x != "N/A" else x)}
            {get_metric_rows("GPU Memory Used", "gpu_memory_used", lambda x: f"{float(x):.1f} GB" if x != "N/A" else x)}
            {get_metric_rows("GPU Utilization", "gpu_utilization", lambda x: f"{float(x):.1f}%" if x != "N/A" else x)}
        </table>
    </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(f"{report_filename}.html", "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"\n==== Model Comparison Report ====")
    print(f"Report saved to: {report_filename}.html")
    print("\nSummary of compared models:")
    for model in model_names:
        metrics = all_metrics[model]
        print(f"\n{model}:")
        print(f"  Embedding Time: {metrics.get('embedding_time', 'N/A'):.2f} seconds")
        print(f"  F1 Score: {metrics.get('f1', 'N/A'):.3f}")
        print(f"  Embeddings/Second: {metrics.get('embeddings_per_second', 'N/A')}")
