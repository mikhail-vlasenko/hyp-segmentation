from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import glob
from datetime import datetime

app = FastAPI(title="Embeddings Viewer", description="Interactive viewer for hierarchy embeddings")

# Path to the spin_dataset directory
SPIN_DATASET_PATH = "hierarchies/hierarchy_embeddings/experiments/spin_dataset"


def scan_embeddings_data() -> List[Dict]:
    """Scan the spin_dataset directory and extract metadata from all experiments."""
    embeddings_data = []
    
    # Threshold for initialization type classification
    tree_aware_threshold = "2025-06-09_180307"

    # Find all config.json files
    pattern = os.path.join(SPIN_DATASET_PATH, "*", "*", "config.json")
    config_files = glob.glob(pattern)

    for config_path in config_files:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Extract directory info
            path_parts = Path(config_path).parts
            hierarchy_dir = path_parts[-3]
            timestamp_dir = path_parts[-2]
            experiment_dir = os.path.dirname(config_path)

            # Get file sizes and modification times
            config_stat = os.stat(config_path)

            # Look for visualization files
            visualizations = []
            for img_file in glob.glob(os.path.join(experiment_dir, "*.png")):
                visualizations.append(os.path.basename(img_file))

            # Determine initialization type based on timestamp
            initialization_type = "tree-aware" if timestamp_dir >= tree_aware_threshold else "random"

            entry = {
                "id": f"{hierarchy_dir}_{timestamp_dir}",
                "hierarchy_name": config.get("hierarchy_name", hierarchy_dir),
                "embedding_dim": config.get("embedding_dim", 0),
                "loss_power": config.get("loss_power", 1.0),
                "timestamp": timestamp_dir,
                "created_at": datetime.fromtimestamp(config_stat.st_mtime).isoformat(),
                "config": config,
                "visualizations": visualizations,
                "path": experiment_dir,
                "epochs": config.get("epochs", 0),
                "lr": config.get("lr", 0),
                "batch_size": config.get("batch_size", 0),
                "curvature": config.get("curvature", 1.0),
                "num_negs": config.get("num_negs", 0),
                "initialization_type": initialization_type
            }
            embeddings_data.append(entry)

        except Exception as e:
            print(f"Error processing {config_path}: {e}")
            continue

    # Sort by timestamp descending (newest first)
    embeddings_data.sort(key=lambda x: x["timestamp"], reverse=True)
    return embeddings_data


@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Embeddings Viewer</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f8f9fa; 
            }
            .container { max-width: 1400px; margin: 0 auto; }
            h1 { color: #2c3e50; margin-bottom: 30px; }
            .filters {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                align-items: center;
            }
            .filter-group {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            .filter-group label {
                font-weight: 600;
                color: #555;
                font-size: 12px;
                text-transform: uppercase;
            }
            input, select {
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
            .table-container {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }
            th {
                background-color: #f8f9fa;
                font-weight: 600;
                color: #555;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            tr:hover {
                background-color: #f8f9fa;
            }
            .timestamp {
                font-family: monospace;
                font-size: 12px;
                color: #666;
            }
            .badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
            }
            .badge-dim {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            .badge-power {
                background-color: #f3e5f5;
                color: #7b1fa2;
            }
            .badge-init-tree {
                background-color: #e8f5e8;
                color: #2e7d32;
            }
            .badge-init-random {
                background-color: #fff3e0;
                color: #f57c00;
            }
            .clickable-row {
                cursor: pointer;
            }
            .clickable-row:hover {
                background-color: #e3f2fd !important;
            }
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.8);
            }
            .modal-content {
                margin: 5% auto;
                padding: 20px;
                background: white;
                border-radius: 8px;
                max-width: 90%;
                max-height: 80%;
                text-align: center;
                overflow: auto;
            }
            .modal-image {
                max-width: 100%;
                max-height: 70vh;
                object-fit: contain;
            }
            .modal-close {
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
                color: #aaa;
            }
            .modal-close:hover {
                color: black;
            }
            .copy-btn {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 4px 8px;
                margin-left: 8px;
                cursor: pointer;
                font-size: 12px;
                color: #6c757d;
                transition: all 0.2s;
            }
            .copy-btn:hover {
                background: #e9ecef;
                color: #495057;
            }
            .copy-btn.copied {
                background: #d1ecf1;
                color: #0c5460;
                border-color: #bee5eb;
            }
            .visualizations {
                display: flex;
                gap: 5px;
                flex-wrap: wrap;
            }
            .viz-item {
                background: #e8f5e8;
                color: #2e7d32;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 10px;
            }
            .loading {
                text-align: center;
                padding: 40px;
                color: #666;
            }
            .stats {
                display: flex;
                gap: 20px;
                margin-bottom: 15px;
                font-size: 14px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Embeddings Viewer</h1>

            <div class="filters">
                <div class="filter-group">
                    <label>Hierarchy Name</label>
                    <select id="hierarchyFilter">
                        <option value="">All Hierarchies</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Embedding Dimension</label>
                    <select id="dimFilter">
                        <option value="">All Dimensions</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Loss Power</label>
                    <select id="powerFilter">
                        <option value="">All Powers</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Search</label>
                    <input type="text" id="searchInput" placeholder="Search experiments...">
                </div>
            </div>

            <div class="stats" id="stats"></div>

            <div class="table-container">
                <div class="loading" id="loading">Loading experiments...</div>
                <table id="experimentsTable" style="display: none;">
                    <thead>
                        <tr>
                            <th>Experiment</th>
                            <th>Hierarchy</th>
                            <th>Dim</th>
                            <th>Loss Power</th>
                            <th>Initialization</th>
                            <th>Config</th>
                            <th>Visualizations</th>
                            <th>Created</th>
                        </tr>
                    </thead>
                    <tbody id="experimentsBody">
                    </tbody>
                </table>
            </div>

            <!-- Modal for displaying images -->
            <div id="imageModal" class="modal">
                <div class="modal-content">
                    <span class="modal-close">&times;</span>
                    <h3 id="modalTitle">Prototype Edge Distortions</h3>
                    <img id="modalImage" class="modal-image" src="" alt="Prototype Edge Distortions">
                </div>
            </div>
        </div>

        <script>
            let allData = [];
            let filteredData = [];

            async function loadData() {
                try {
                    const response = await fetch('/api/embeddings');
                    allData = await response.json();
                    filteredData = [...allData];
                    populateFilters();
                    renderTable();
                    updateStats();
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('experimentsTable').style.display = 'table';
                } catch (error) {
                    console.error('Error loading data:', error);
                    document.getElementById('loading').textContent = 'Error loading data';
                }
            }

            function populateFilters() {
                const hierarchies = [...new Set(allData.map(d => d.hierarchy_name))].sort();
                const dims = [...new Set(allData.map(d => d.embedding_dim))].sort((a, b) => a - b);
                const powers = [...new Set(allData.map(d => d.loss_power))].sort((a, b) => a - b);

                populateSelect('hierarchyFilter', hierarchies);
                populateSelect('dimFilter', dims);
                populateSelect('powerFilter', powers);
            }

            function populateSelect(id, values) {
                const select = document.getElementById(id);
                const currentValue = select.value;
                select.innerHTML = `<option value="">All ${id.replace('Filter', '').replace(/([A-Z])/g, ' $1')}</option>`;
                values.forEach(value => {
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = value;
                    select.appendChild(option);
                });
                select.value = currentValue;
            }

            function applyFilters() {
                const hierarchyFilter = document.getElementById('hierarchyFilter').value;
                const dimFilter = document.getElementById('dimFilter').value;
                const powerFilter = document.getElementById('powerFilter').value;
                const searchFilter = document.getElementById('searchInput').value.toLowerCase();

                filteredData = allData.filter(item => {
                    return (!hierarchyFilter || item.hierarchy_name === hierarchyFilter) &&
                           (!dimFilter || item.embedding_dim == dimFilter) &&
                           (!powerFilter || item.loss_power == powerFilter) &&
                           (!searchFilter || 
                            item.hierarchy_name.toLowerCase().includes(searchFilter) ||
                            item.timestamp.toLowerCase().includes(searchFilter) ||
                            item.id.toLowerCase().includes(searchFilter));
                });

                renderTable();
                updateStats();
            }

            function renderTable() {
                const tbody = document.getElementById('experimentsBody');
                tbody.innerHTML = '';

                filteredData.forEach(item => {
                    const row = document.createElement('tr');
                    row.className = 'clickable-row';
                    row.dataset.experimentId = item.id;

                    const configDetails = `Epochs: ${item.epochs}, LR: ${item.lr}, Batch: ${item.batch_size}, Curvature: ${item.curvature}`;
                    const initBadgeClass = item.initialization_type === 'tree-aware' ? 'badge-init-tree' : 'badge-init-random';

                    row.innerHTML = `
                        <td>
                            <div style="font-weight: 600;">${item.id}</div>
                            <div class="timestamp">
                                ${item.timestamp}
                                <button class="copy-btn" onclick="copyToClipboard('${item.timestamp}', this); event.stopPropagation();" title="Copy date to clipboard">📋</button>
                            </div>
                        </td>
                        <td>${item.hierarchy_name}</td>
                        <td><span class="badge badge-dim">${item.embedding_dim}D</span></td>
                        <td><span class="badge badge-power">${item.loss_power}</span></td>
                        <td><span class="badge ${initBadgeClass}">${item.initialization_type}</span></td>
                        <td title="${configDetails}">${configDetails}</td>
                        <td>
                            <div class="visualizations">
                                ${item.visualizations.map(viz => `<span class="viz-item">${viz.replace('.png', '')}</span>`).join('')}
                            </div>
                        </td>
                        <td class="timestamp">${new Date(item.created_at).toLocaleString()}</td>
                    `;
                    
                    // Add click event listener
                    row.addEventListener('click', () => showImageModal(item));
                    
                    tbody.appendChild(row);
                });
            }

            function updateStats() {
                const stats = document.getElementById('stats');
                const totalExperiments = allData.length;
                const filteredExperiments = filteredData.length;
                const uniqueHierarchies = new Set(filteredData.map(d => d.hierarchy_name)).size;
                const uniqueDims = new Set(filteredData.map(d => d.embedding_dim)).size;

                stats.innerHTML = `
                    <span>📊 ${filteredExperiments} of ${totalExperiments} experiments</span>
                    <span>🏗️ ${uniqueHierarchies} hierarchies</span>
                    <span>📐 ${uniqueDims} dimensions</span>
                `;
            }

            function showImageModal(item) {
                const modal = document.getElementById('imageModal');
                const modalImage = document.getElementById('modalImage');
                const modalTitle = document.getElementById('modalTitle');
                
                modalTitle.textContent = `${item.id} - Prototype Edge Distortions`;
                modalImage.src = `/api/image/${item.id}/prototype_edge_distortions.png`;
                modal.style.display = 'block';
            }

            function closeModal() {
                document.getElementById('imageModal').style.display = 'none';
            }

            function copyToClipboard(text, button) {
                navigator.clipboard.writeText(text).then(() => {
                    // Visual feedback
                    const originalText = button.textContent;
                    button.textContent = '✓';
                    button.classList.add('copied');
                    
                    setTimeout(() => {
                        button.textContent = originalText;
                        button.classList.remove('copied');
                    }, 1500);
                }).catch(err => {
                    console.error('Failed to copy: ', err);
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = text;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    
                    // Visual feedback for fallback
                    const originalText = button.textContent;
                    button.textContent = '✓';
                    button.classList.add('copied');
                    
                    setTimeout(() => {
                        button.textContent = originalText;
                        button.classList.remove('copied');
                    }, 1500);
                });
            }

            // Event listeners
            document.getElementById('hierarchyFilter').addEventListener('change', applyFilters);
            document.getElementById('dimFilter').addEventListener('change', applyFilters);
            document.getElementById('powerFilter').addEventListener('change', applyFilters);
            document.getElementById('searchInput').addEventListener('input', applyFilters);
            
            // Modal event listeners
            document.querySelector('.modal-close').addEventListener('click', closeModal);
            document.getElementById('imageModal').addEventListener('click', (e) => {
                if (e.target === e.currentTarget) {
                    closeModal();
                }
            });

            // Load data on page load
            loadData();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/embeddings")
async def get_embeddings() -> List[Dict]:
    """Get all embeddings data with metadata."""
    try:
        return scan_embeddings_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scanning embeddings: {str(e)}")


@app.get("/api/embeddings/{experiment_id}")
async def get_embedding_details(experiment_id: str) -> Dict:
    """Get detailed information about a specific experiment."""
    try:
        embeddings_data = scan_embeddings_data()
        experiment = next((item for item in embeddings_data if item["id"] == experiment_id), None)

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        return experiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting experiment details: {str(e)}")


@app.get("/api/image/{experiment_id}/{image_name}")
async def get_experiment_image(experiment_id: str, image_name: str):
    """Serve an image file from a specific experiment."""
    try:
        embeddings_data = scan_embeddings_data()
        experiment = next((item for item in embeddings_data if item["id"] == experiment_id), None)
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        image_path = os.path.join(experiment["path"], image_name)
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8002)
