// document.getElementById('upload-input').addEventListener('change', handleFileUpload);
document.getElementById('upload-btn').addEventListener('click', handleUploadClick);
document.getElementById('retry-btn').addEventListener('click', resetPage);

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('dog-image').src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
}

function handleUploadClick() {
    const fileInput = document.getElementById('upload-input');
    const file = fileInput.files[0];
    if (!file) {
        alert("Please upload a file first.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Send the file to the server
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById('result-section').classList.remove('hidden');
        document.getElementById('breed-name').textContent = data.predicted_breed;
        document.getElementById('confidence').textContent = data.confidence_score;
    })
    .catch(error => console.error('Error:', error));
}

function resetPage() {
    document.getElementById('result-section').classList.add('hidden');
    document.getElementById('upload-input').value = '';
    document.getElementById('dog-image').src = '#';
}