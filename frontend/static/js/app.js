// Cybersecurity ML Framework - Frontend JavaScript

// Global variables
let currentDataset = null;
let trainedModels = {};
let currentResults = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    updateStatus();
    setInterval(updateStatus, 5000); // Update status every 5 seconds
});

// Utility functions
function showLoading() {
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alert-container');
    const alertId = 'alert-' + Date.now();
    
    const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" id="${alertId}" role="alert">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-circle' : 'info-circle'}"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    alertContainer.innerHTML = alertHtml;
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = document.getElementById(alertId);
        if (alert) {
            alert.remove();
        }
    }, 5000);
}

function updateStatus() {
    fetch('/api/get_status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('models-count').textContent = data.models_trained.length;
            document.getElementById('datasets-count').textContent = data.datasets_available.length;
            document.getElementById('results-count').textContent = data.results_available.length;
            
            // Update model dropdowns
            updateModelDropdowns(data.models_trained);
            
            // Update system status
            const status = data.models_trained.length > 0 ? 'Active' : 'Ready';
            document.getElementById('system-status').textContent = status;
        })
        .catch(error => {
            console.error('Error updating status:', error);
        });
}

function updateModelDropdowns(models) {
    const predictionModel = document.getElementById('prediction-model');
    const visualizationModel = document.getElementById('visualization-model');
    
    // Clear existing options except the first one
    predictionModel.innerHTML = '<option value="">Select a trained model</option>';
    visualizationModel.innerHTML = '<option value="">Select a trained model</option>';
    
    // Add model options
    models.forEach(model => {
        const option1 = document.createElement('option');
        option1.value = model;
        option1.textContent = model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        predictionModel.appendChild(option1);
        
        const option2 = document.createElement('option');
        option2.value = model;
        option2.textContent = model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
        visualizationModel.appendChild(option2);
    });
}

// Data generation
function generateData() {
    showLoading();
    
    const nSamples = parseInt(document.getElementById('n-samples').value);
    const nFeatures = parseInt(document.getElementById('n-features').value);
    const nClasses = parseInt(document.getElementById('n-classes').value);
    
    const data = {
        n_samples: nSamples,
        n_features: nFeatures,
        n_classes: nClasses
    };
    
    fetch('/api/generate_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideLoading();
        if (result.success) {
            currentDataset = result.dataset_info;
            showAlert(`Dataset generated successfully! ${result.dataset_info.n_samples} samples with ${result.dataset_info.n_features} features.`, 'success');
            updateStatus();
        } else {
            showAlert(`Error: ${result.error}`, 'danger');
        }
    })
    .catch(error => {
        hideLoading();
        showAlert(`Error generating data: ${error.message}`, 'danger');
    });
}

// Model training
function trainModel() {
    if (!currentDataset) {
        showAlert('Please generate a dataset first!', 'warning');
        return;
    }
    
    showLoading();
    
    const modelType = document.getElementById('model-type').value;
    const nEstimators = parseInt(document.getElementById('n-estimators').value);
    const maxDepth = parseInt(document.getElementById('max-depth').value);
    
    const data = {
        model_type: modelType,
        params: {
            n_estimators: nEstimators,
            max_depth: maxDepth
        }
    };
    
    fetch('/api/train_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideLoading();
        if (result.success) {
            trainedModels[modelType] = result.results;
            showAlert(`${modelType.replace('_', ' ').toUpperCase()} model trained successfully! Accuracy: ${(result.results.accuracy * 100).toFixed(2)}%`, 'success');
            displayResults(modelType, result.results);
            updateStatus();
        } else {
            showAlert(`Error: ${result.error}`, 'danger');
        }
    })
    .catch(error => {
        hideLoading();
        showAlert(`Error training model: ${error.message}`, 'danger');
    });
}

// Anomaly detector training
function trainAnomalyDetector() {
    if (!currentDataset) {
        showAlert('Please generate a dataset first!', 'warning');
        return;
    }
    
    showLoading();
    
    const detectorType = document.getElementById('detector-type').value;
    const contamination = parseFloat(document.getElementById('contamination').value);
    
    const data = {
        detector_type: detectorType,
        params: {
            contamination: contamination
        }
    };
    
    fetch('/api/train_anomaly_detector', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideLoading();
        if (result.success) {
            trainedModels[`anomaly_${detectorType}`] = result.results;
            showAlert(`${detectorType.replace('_', ' ').toUpperCase()} detector trained successfully! Accuracy: ${(result.results.accuracy * 100).toFixed(2)}%`, 'success');
            displayAnomalyResults(`anomaly_${detectorType}`, result.results);
            updateStatus();
        } else {
            showAlert(`Error: ${result.error}`, 'danger');
        }
    })
    .catch(error => {
        hideLoading();
        showAlert(`Error training detector: ${error.message}`, 'danger');
    });
}

// Make prediction
function makePrediction() {
    const modelType = document.getElementById('prediction-model').value;
    const inputData = document.getElementById('input-data').value;
    
    if (!modelType) {
        showAlert('Please select a trained model!', 'warning');
        return;
    }
    
    if (!inputData.trim()) {
        showAlert('Please enter input data!', 'warning');
        return;
    }
    
    showLoading();
    
    // Parse input data
    const features = inputData.split(',').map(x => parseFloat(x.trim()));
    
    if (features.some(isNaN)) {
        hideLoading();
        showAlert('Please enter valid numeric values separated by commas!', 'danger');
        return;
    }
    
    const data = {
        model_type: modelType,
        input_data: features
    };
    
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideLoading();
        if (result.success) {
            const prediction = result.prediction;
            const probability = result.probability;
            
            let predictionText = '';
            if (probability && probability.length > 1) {
                predictionText = `Prediction: ${prediction} (Confidence: ${(probability[prediction] * 100).toFixed(2)}%)`;
            } else {
                predictionText = `Prediction: ${prediction}`;
            }
            
            showAlert(predictionText, 'success');
        } else {
            showAlert(`Error: ${result.error}`, 'danger');
        }
    })
    .catch(error => {
        hideLoading();
        showAlert(`Error making prediction: ${error.message}`, 'danger');
    });
}

// Generate visualization
function generateVisualization() {
    const plotType = document.getElementById('plot-type').value;
    const modelType = document.getElementById('visualization-model').value;
    
    if (!modelType) {
        showAlert('Please select a trained model!', 'warning');
        return;
    }
    
    showLoading();
    
    const data = {
        plot_type: plotType,
        model_type: modelType
    };
    
    fetch('/api/visualize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        hideLoading();
        if (result.success) {
            const plotImage = document.getElementById('plot-image');
            const container = document.getElementById('visualization-container');
            
            plotImage.src = 'data:image/png;base64,' + result.image;
            container.style.display = 'block';
            
            showAlert(`${plotType.replace('_', ' ').toUpperCase()} visualization generated successfully!`, 'success');
        } else {
            showAlert(`Error: ${result.error}`, 'danger');
        }
    })
    .catch(error => {
        hideLoading();
        showAlert(`Error generating visualization: ${error.message}`, 'danger');
    });
}

// Display results
function displayResults(modelType, results) {
    const container = document.getElementById('results-container');
    
    const resultsHtml = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <div class="metric-value">${(results.accuracy * 100).toFixed(2)}%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <div class="metric-value">${results.confusion_matrix[0][0] + results.confusion_matrix[1][1]}</div>
                    <div class="metric-label">Correct Predictions</div>
                </div>
            </div>
        </div>
        
        <div class="mt-3">
            <h5>Model: ${modelType.replace('_', ' ').toUpperCase()}</h5>
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Accuracy</td>
                            <td>${(results.accuracy * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>ROC AUC</td>
                            <td>${results.roc_auc ? results.roc_auc.toFixed(3) : 'N/A'}</td>
                        </tr>
                        <tr>
                            <td>True Positives</td>
                            <td>${results.confusion_matrix[1][1]}</td>
                        </tr>
                        <tr>
                            <td>True Negatives</td>
                            <td>${results.confusion_matrix[0][0]}</td>
                        </tr>
                        <tr>
                            <td>False Positives</td>
                            <td>${results.confusion_matrix[0][1]}</td>
                        </tr>
                        <tr>
                            <td>False Negatives</td>
                            <td>${results.confusion_matrix[1][0]}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    container.innerHTML = resultsHtml;
}

// Display anomaly detection results
function displayAnomalyResults(detectorType, results) {
    const container = document.getElementById('results-container');
    
    const resultsHtml = `
        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <div class="metric-value">${(results.accuracy * 100).toFixed(2)}%</div>
                    <div class="metric-label">Detection Accuracy</div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <div class="metric-value">${results.anomalies_detected}</div>
                    <div class="metric-label">Anomalies Detected</div>
                </div>
            </div>
        </div>
        
        <div class="mt-3">
            <h5>Detector: ${detectorType.replace('anomaly_', '').replace('_', ' ').toUpperCase()}</h5>
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Detection Accuracy</td>
                            <td>${(results.accuracy * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>Anomalies Detected</td>
                            <td>${results.anomalies_detected}</td>
                        </tr>
                        <tr>
                            <td>Total Samples</td>
                            <td>${results.predictions.length}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    container.innerHTML = resultsHtml;
}

// Auto-generate sample input data
function generateSampleInput() {
    if (currentDataset) {
        const nFeatures = currentDataset.n_features;
        const sampleData = Array.from({length: nFeatures}, () => (Math.random() * 10 - 5).toFixed(2));
        document.getElementById('input-data').value = sampleData.join(', ');
    } else {
        showAlert('Please generate a dataset first!', 'warning');
    }
}

// Add event listeners for better UX
document.addEventListener('DOMContentLoaded', function() {
    // Add sample data generation button
    const predictionCard = document.querySelector('.card:has(.btn-info)');
    if (predictionCard) {
        const button = document.createElement('button');
        button.className = 'btn btn-outline-secondary btn-sm mt-2';
        button.innerHTML = '<i class="fas fa-random"></i> Generate Sample Data';
        button.onclick = generateSampleInput;
        predictionCard.querySelector('.card-body').appendChild(button);
    }
});
