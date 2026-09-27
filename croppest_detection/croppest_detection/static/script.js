document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const uploadArea = document.getElementById('upload-container');
    const fileInput = document.getElementById('file-input');
    
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const resetBtn = document.getElementById('reset-btn');
    const predictBtn = document.getElementById('predict-btn');
    
    const loading = document.getElementById('loading');
    
    const resultContainer = document.getElementById('result-container');
    const successCard = document.getElementById('success-card');
    const pestName = document.getElementById('pest-name');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    
    const errorContainer = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');

    let currentFile = null;

    // --- Drag and Drop Events ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        }, false);
    });

    uploadArea.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // --- File Input Change ---
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // --- Core Logic ---
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG).');
            return;
        }
        
        currentFile = file;
        
        // Render Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            
            // Switch views
            uploadArea.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultContainer.classList.add('hidden');
            errorContainer.classList.add('hidden');
            successCard.style.display = 'flex'; // Reset in case it was hidden by error
            
            // Reset Progress Bar state
            confidenceBar.style.width = '0%';
            confidenceText.innerText = '0.00%';
        };
        reader.readAsDataURL(file);
    }

    // --- Clear & Reset ---
    resetBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        
        previewContainer.classList.add('hidden');
        resultContainer.classList.add('hidden');
        loading.classList.add('hidden');
        
        uploadArea.classList.remove('hidden');
    });

    // --- Predict / Upload to API ---
    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI toggles
        predictBtn.disabled = true;
        resetBtn.disabled = true;
        loading.classList.remove('hidden');
        
        // Hide previous results during loading
        resultContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            // Initiate Fast API POST request
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();

            if (data.success) {
                showSuccess(data.prediction, data.confidence);
            } else {
                showError(data.error || "An unknown error from the server.");
            }

        } catch (err) {
            console.error("Inference fetch error:", err);
            showError("Cannot connect to backend server. Make sure it is running via uvicorn.");
        } finally {
            predictBtn.disabled = false;
            resetBtn.disabled = false;
            loading.classList.add('hidden');
        }
    });

    // --- Renderers ---
    function showSuccess(predictedClass, confidenceScoreStr) {
        pestName.innerText = predictedClass;
        confidenceText.innerText = confidenceScoreStr;
        
        // Extract numeric part for the bar
        const confNumber = confidenceScoreStr.replace('%', '');
        
        errorContainer.classList.add('hidden');
        successCard.style.display = 'flex';
        resultContainer.classList.remove('hidden');

        // Delay animation slightly so browser renders DOM update first
        setTimeout(() => {
            confidenceBar.style.width = `${confNumber}%`;
        }, 50);
    }

    function showError(msg) {
        errorText.innerText = msg;
        successCard.style.display = 'none';
        errorContainer.classList.remove('hidden');
        resultContainer.classList.remove('hidden');
    }
});
