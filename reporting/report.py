def print_report(model_name, ds, db_type, host, port, collection, embedding_time, retrieval_time, top_k, metrics):
    import torch
    import datetime
    import os
    # Gather additional details
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    data_path = ds.get('file', ds.get('table', ''))
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 'N/A'
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # HTML Report
    html = f"""
    <html>
    <head>
    <style>
    body {{ font-family: Arial, sans-serif; background: #f8f8f8; }}
    .report {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #ccc; padding: 32px; margin: 32px auto; max-width: 900px; }}
    h2 {{ color: #2c3e50; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }}
    th {{ background: #f0f0f0; }}
    .section {{ margin-top: 32px; }}
    .summary {{ background: #eaf6ff; padding: 16px; border-radius: 6px; margin-bottom: 24px; }}
    .metric {{ font-weight: bold; color: #2980b9; }}
    </style>
    </head>
    <body>
    <div class="report">
    <h2>Embedding Model Performance Report</h2>
    <div class="summary">
      <b>Run Time:</b> {now}<br>
      <b>Embedding Model:</b> {model_name}<br>
      <b>Model Config:</b> <span class="metric">Dimension: {metrics.get('dimension', 'N/A')}, Batch Size: {metrics.get('batch_size', 'N/A')}, Parallelism: {metrics.get('parallelism', 'N/A')}</span><br>
      <b>Data Source:</b> {ds['type']} ({data_path})<br>
      <b>Data Size:</b> {data_size} bytes<br>
      <b>Vector DB:</b> {db_type} ({host}:{port})<br>
      <b>Collection:</b> {collection}<br>
      <b>PyTorch Device:</b> {device} ({device_name})<br>
    </div>
    <div class="section">
      <h3>Embedding Generation</h3>
      <table>
        <tr><th>Time Taken</th><td>{metrics.get('embedding_time', 'N/A')} seconds</td></tr>
        <tr><th>Total Embeddings</th><td>{metrics.get('total_embeddings', 'N/A')}</td></tr>
        <tr><th>Resources Used</th><td>{device_name}</td></tr>
      </table>
    </div>
    <div class="section">
      <h3>Vector DB Insertion</h3>
      <table>
        <tr><th>Insertion Time</th><td>{metrics.get('insertion_time', 'N/A')} seconds</td></tr>
        <tr><th>Batch Size</th><td>{metrics.get('batch_size', 'N/A')}</td></tr>
        <tr><th>Retries</th><td>{metrics.get('upsert_retries', 'N/A')}</td></tr>
      </table>
    </div>
    <div class="section">
      <h3>Retrieval & Evaluation</h3>
      <table>
        <tr><th>Retrieval Time</th><td>{metrics.get('retrieval_time', 'N/A')} seconds</td></tr>
        <tr><th>Top-K</th><td>{metrics.get('top_k', 'N/A')}</td></tr>
        <tr><th>Accuracy</th><td>{metrics['accuracy']:.3f}</td></tr>
        <tr><th>Recall</th><td>{metrics['recall']:.3f}</td></tr>
        <tr><th>Precision</th><td>{metrics['precision']:.3f}</td></tr>
        <tr><th>F1 Score</th><td>{metrics['f1']:.3f}</td></tr>
      </table>
    </div>
    <div class="section">
      <h3>Summary & Insights</h3>
      <ul>
        <li>Model <b>{model_name}</b> with dimension <b>{metrics.get('dimension', 'N/A')}</b> processed <b>{metrics.get('total_embeddings', 'N/A')}</b> embeddings in <b>{metrics.get('embedding_time', 'N/A')}</b> seconds.</li>
        <li>Data source <b>{data_path}</b> ({ds['type']}) size: <b>{data_size} bytes</b>.</li>
        <li>Embeddings inserted into <b>{db_type}</b> collection <b>{collection}</b> in <b>{metrics.get('insertion_time', 'N/A')}</b> seconds (batch size: <b>{metrics.get('batch_size', 'N/A')}</b>, retries: <b>{metrics.get('upsert_retries', 'N/A')}</b>).</li>
        <li>Evaluation metrics: Accuracy <b>{metrics['accuracy']:.3f}</b>, Recall <b>{metrics['recall']:.3f}</b>, Precision <b>{metrics['precision']:.3f}</b>, F1 <b>{metrics['f1']:.3f}</b>.</li>
      </ul>
    </div>
    </div>
    </body>
    </html>
    """
    # Save HTML report
    with open("embedding_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("\n==== Embedding Model Performance Report ====")
    print("Report saved to embedding_report.html")
    print("Summary:")
    print(f"Embedding Model: {model_name}")
    print(f"Data Source: {ds['type']} ({data_path}) Size: {data_size} bytes")
    print(f"Vector DB: {db_type} ({host}:{port}) Collection: {collection}")
    print(f"PyTorch Device: {device} ({device_name})")
    print(f"Embedding Time: {metrics.get('embedding_time', embedding_time):.2f} seconds")
    print(f"Total Embeddings: {metrics.get('total_embeddings', 'N/A')}")
    print(f"Insertion Time: {metrics.get('insertion_time', 'N/A')} seconds")
    print(f"Batch Size: {metrics.get('batch_size', 'N/A')}")
    print(f"Retries: {metrics.get('upsert_retries', 'N/A')}")
    print(f"Retrieval Time: {metrics.get('retrieval_time', retrieval_time):.2f} seconds")
    print(f"Top-K: {metrics.get('top_k', top_k)}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
