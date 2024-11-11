from flask import Flask, request, jsonify, render_template
import os
import requests  # Import the requests library to make API calls

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file with a consistent name
    if file:
        # Extract the file extension
        _, file_extension = os.path.splitext(file.filename)
        
        # Define the new filename
        new_filename = f"prediction_image.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        
        # Save the file with the new name
        file.save(file_path)
        
        # Now call the external API to process the image and return the file
        external_api_url = 'http://127.0.0.1:5500/getPredictionOutput'  # Replace with the actual API URL
        with open(file_path, 'rb') as image_file:
            files = {'file': (new_filename, image_file, 'image/jpeg')}
            try:
                external_api_response = requests.get(external_api_url, files=files)
                external_api_response.raise_for_status()  # Raise an error if the response was not successful
                # Assuming the external API returns a URL to the processed image
                processed_image_url = external_api_response.json().get('processed_image_url')
            except requests.exceptions.RequestException as e:
                return jsonify({'error': f'Error calling external API: {str(e)}'}), 500

        # Return the URL of the processed image to the client
        return jsonify({
            'message': 'File uploaded and processed successfully!',
            'original_image_url': file_path,
            'processed_image_url': processed_image_url
        }), 200

if __name__ == '__main__':
    app.run(debug=True, port=6000)