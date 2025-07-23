# Model Comparison Report Example

Below is an example of the HTML report generated for model comparison:

```html
<div class="report">
    <h2>Embedding Model Comparison Report</h2>
    
    <!-- System Configuration -->
    <h3>1. System Configuration</h3>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>CPU Model</td>
            <td>Intel(R) Core(TM) i7-1165G7</td>
        </tr>
        <tr>
            <td>GPU Device</td>
            <td>NVIDIA RTX 3060</td>
        </tr>
    </table>

    <!-- Model Performance -->
    <h3>2. Performance Comparison</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>all-MiniLM-L6-v2</th>
            <th>all-mpnet-base-v2</th>
        </tr>
        <tr>
            <td>Embeddings/Second</td>
            <td>1250.5</td>
            <td>980.2</td>
        </tr>
        <tr>
            <td>F1 Score</td>
            <td>0.856</td>
            <td>0.892</td>
        </tr>
    </table>
</div>
```

The report includes:
- System configuration and resources
- Dataset details
- Side-by-side model performance metrics
- Quality comparison (Precision, Recall, F1)
- Resource utilization comparison
- Processing speed comparison
