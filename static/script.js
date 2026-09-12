// --- Navigation Logic ---
function scrollToUpload() {
    const landingView = document.getElementById('landingView');
    const toolView = document.getElementById('toolView');
    
    // Hide landing view elements (optional: fade them out)
    landingView.style.opacity = '0';
    setTimeout(() => {
        landingView.style.display = 'none';
        
        // Show tool view
        toolView.style.display = 'block';
        setTimeout(() => {
            toolView.style.opacity = '1';
            // Scroll to it
            toolView.scrollIntoView({ behavior: 'smooth' });
        }, 50);
    }, 400);
}

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const controlsSection = document.getElementById('controlsSection');
    const fileCountText = document.getElementById('fileCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loadingState = document.getElementById('loadingState');
    const resultsSection = document.getElementById('resultsSection');
    const resultsGrid = document.getElementById('resultsGrid');
    const passCount = document.getElementById('passCount');
    const failCount = document.getElementById('failCount');

    let selectedFiles = [];

    // --- Drag and Drop Logic ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        handleFiles(dt.files);
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        selectedFiles = Array.from(files);
        if (selectedFiles.length > 0) {
            fileCountText.textContent = `${selectedFiles.length} file${selectedFiles.length > 1 ? 's' : ''} selected`;
            controlsSection.style.display = 'flex';
            resultsSection.style.display = 'none';
        }
    }

    // --- Actions ---
    clearBtn.addEventListener('click', () => {
        selectedFiles = [];
        fileInput.value = '';
        controlsSection.style.display = 'none';
        resultsSection.style.display = 'none';
    });

    analyzeBtn.addEventListener('click', async () => {
        if (selectedFiles.length === 0) return;

        // UI State update
        controlsSection.style.display = 'none';
        loadingState.style.display = 'block';
        resultsSection.style.display = 'none';
        resultsGrid.innerHTML = '';
        
        let pass = 0;
        let fail = 0;

        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('images', file);
        });

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            data.results.forEach((result, index) => {
                if (result.error) {
                    // Handle error card
                    createErrorCard(result.filename, result.error);
                } else {
                    // Update stats
                    if (result.is_defect) {
                        fail++;
                    } else {
                        pass++;
                    }
                    createResultCard(result, index);
                }
            });

            // Update header stats
            passCount.textContent = `${pass} Pass`;
            failCount.textContent = `${fail} Defect${fail !== 1 ? 's' : ''}`;
            
        } catch (error) {
            console.error('Error analyzing images:', error);
            alert('Failed to analyze images. Ensure the model is trained and server is running.');
        } finally {
            loadingState.style.display = 'none';
            controlsSection.style.display = 'flex';
            resultsSection.style.display = 'block';
        }
    });

    function createResultCard(result, index) {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.style.animationDelay = `${index * 0.1}s`;

        const statusClass = result.is_defect ? 'status-defect' : 'status-pass';
        const statusText = result.is_defect ? 'DEFECT' : 'PASS';
        const confidencePct = (result.confidence * 100).toFixed(1) + '%';

        card.innerHTML = `
            <div class="card-header">
                <span class="filename" title="${result.filename}">${result.filename}</span>
                <span class="status-badge ${statusClass}">${statusText}</span>
            </div>
            <div class="image-container">
                <img src="${result.original_image}" alt="Original" class="img-original">
                ${result.heatmap_image ? `<img src="${result.heatmap_image}" alt="Heatmap" class="img-heatmap">` : ''}
                <div class="view-hint">Hover for Heatmap</div>
            </div>
            <div class="card-footer">
                <span class="conf-label">Confidence</span>
                <span class="conf-value">${confidencePct}</span>
            </div>
        `;

        resultsGrid.appendChild(card);
    }

    function createErrorCard(filename, error) {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <div class="card-header">
                <span class="filename">${filename}</span>
                <span class="status-badge status-defect">ERROR</span>
            </div>
            <div style="padding: 2rem; text-align: center; color: var(--danger);">
                <i class="fa-solid fa-triangle-exclamation" style="font-size: 2rem; margin-bottom: 1rem;"></i>
                <p style="font-size: 0.9rem;">${error}</p>
            </div>
        `;
        resultsGrid.appendChild(card);
    }
});
