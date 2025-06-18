document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const predictButton = document.getElementById('predictButton');
    const uploadedImage = document.getElementById('uploadedImage');
    const previewText = document.getElementById('previewText');
    const resultsSection = document.getElementById('resultsSection');
    const predictedClassSpan = document.getElementById('predictedClass');
    const confidenceSpan = document.getElementById('confidence');
    const probabilitiesList = document.getElementById('probabilitiesList');
    const errorMessageDiv = document.getElementById('errorMessage');
    const errorDetailsSpan = document.getElementById('errorDetails');
    const loadingSpinner = document.getElementById('loadingSpinner');

    let selectedFile = null;

    // Handle file selection
    imageUpload.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = (e) => {
                uploadedImage.src = e.target.result;
                uploadedImage.style.display = 'block';
                previewText.style.display = 'none';
            };
            reader.readAsDataURL(selectedFile);
            predictButton.disabled = false; // Enable predict button
            hideResultsAndError(); // Hide previous results/errors
        } else {
            uploadedImage.src = '#';
            uploadedImage.style.display = 'none';
            previewText.style.display = 'block';
            predictButton.disabled = true; // Disable predict button
            hideResultsAndError();
        }
    });

    // Handle predict button click
    predictButton.addEventListener('click', async () => {
        if (!selectedFile) {
            alert('Please select an image first.');
            return;
        }

        hideResultsAndError();
        showLoading();

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict_bird/', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            displayResults(result);

        } catch (error) {
            console.error('Prediction failed:', error);
            showError(error.message || 'An unknown error occurred.');
        } finally {
            hideLoading();
        }
    });

    function displayResults(data) {
        predictedClassSpan.textContent = data.predicted_class;
        confidenceSpan.textContent = (data.confidence * 100).toFixed(2) + '%';

        probabilitiesList.innerHTML = ''; // Clear previous probabilities
        // Sort probabilities from highest to lowest for better readability
        const sortedProbabilities = Object.entries(data.all_probabilities).sort(([, a], [, b]) => b - a);

        sortedProbabilities.forEach(([label, prob]) => {
            const listItem = document.createElement('li');
            listItem.innerHTML = `${label}: <span>${(prob * 100).toFixed(2)}%</span>`;
            probabilitiesList.appendChild(listItem);
        });

        resultsSection.style.display = 'block';
    }

    function hideResultsAndError() {
        resultsSection.style.display = 'none';
        errorMessageDiv.style.display = 'none';
    }

    function showError(message) {
        errorMessageDiv.style.display = 'block';
        errorDetailsSpan.textContent = message;
    }

    function showLoading() {
        loadingSpinner.style.display = 'flex'; // Use flex to center spinner content
    }

    function hideLoading() {
        loadingSpinner.style.display = 'none';
    }
});
