from flask import Flask, request, render_template, jsonify, send_from_directory
import base64
from io import BytesIO
import os
import sys
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ===========================
# Model Class
# ===========================
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# Initialize Flask app
app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB limit

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainTumorCNN(num_classes=2).to(device)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_brain_tumor_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Class labels
class_names = ["No Tumor", "Tumor"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'status': 'error'})
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'})
    
    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PNG or JPEG image.', 'status': 'error'})
    
    try:
        # Read and validate image
        image = Image.open(file.stream).convert('RGB')
        
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        image.save(filename)
        
        # Prepare image for model
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, 1)
            predicted_label = class_names[prediction.item()]
            predicted_conf = confidence.item() * 100
        
        # Prepare base image for display
        buffered = BytesIO()
        display_image = image.copy()
        display_image.thumbnail((300, 300))
        display_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response_data = {
            'prediction': predicted_label,
            'confidence': round(predicted_conf, 2),
            'image': img_str,
            'status': 'success'
        }
        
        # Generate Grad-CAM visualization only if tumor is detected
        if predicted_label == "Tumor":
            # Initialize Grad-CAM with the last convolutional layer
            grad_cam = GradCAM(model, model.conv_layers[-2])  # Using the last conv layer before ReLU
            
            # Generate heatmap
            heatmap = grad_cam.generate_heatmap(input_tensor, target_class=1)  # 1 is the tumor class
            
            # Resize heatmap to match original image size
            heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
            
            # Convert heatmap to RGB
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Convert original image to numpy array
            original_img = np.array(image)
            
            # Create overlay
            overlay = cv2.addWeighted(original_img, 0.7, heatmap_colored, 0.3, 0)
            
            # Convert heatmap and overlay to base64
            heatmap_img = Image.fromarray(heatmap_colored)
            overlay_img = Image.fromarray(overlay)
            
            # Resize for display
            heatmap_img.thumbnail((300, 300))
            overlay_img.thumbnail((300, 300))
            
            # Convert to base64
            buffered_heatmap = BytesIO()
            buffered_overlay = BytesIO()
            
            heatmap_img.save(buffered_heatmap, format="JPEG")
            overlay_img.save(buffered_overlay, format="JPEG")
            
            heatmap_str = base64.b64encode(buffered_heatmap.getvalue()).decode('utf-8')
            overlay_str = base64.b64encode(buffered_overlay.getvalue()).decode('utf-8')
            
            response_data['heatmap'] = heatmap_str
            response_data['overlay'] = overlay_str
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })

@app.after_request
def add_header(response):
    # Prevent caching of responses
    response.headers['Cache-Control'] = 'no-store'
    return response

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Hook the target layer
        target_layer.register_forward_hook(self.save_features)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_features(self, module, input, output):
        self.features = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_image, target_class):
        # Forward pass
        output = self.model(input_image)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Get the score for the target class
        score = output[0][target_class]
        
        # Backward pass
        score.backward()
        
        # Get gradients and features
        gradients = self.gradients[0]
        features = self.features[0]
        
        # Calculate weights
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Generate cam
        cam = torch.zeros(features.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * features[i]
        
        # Apply ReLU and normalize
        cam = torch.maximum(cam, torch.tensor(0))
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        return cam.cpu().numpy()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
