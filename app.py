import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
from werkzeug.utils import secure_filename
import time

# Configure logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, f'app_{datetime.now().strftime("%Y%m%d")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(image_path, max_size=400):
    try:
        logger.info(f"Loading image from: {image_path}")
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        aspect_ratio = image.size[0] / image.size[1]
        
        if image.size[0] > max_size:
            size = (max_size, int(max_size / aspect_ratio))
            image = image.resize(size, Image.LANCZOS)
        elif image.size[1] > max_size:
            size = (int(max_size * aspect_ratio), max_size)
            image = image.resize(size, Image.LANCZOS)
        
        logger.info(f"Image resized from {original_size} to {image.size}")
            
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise ValueError(f"Error loading image: {str(e)}")

def save_image(tensor, filename):
    try:
        logger.info(f"Saving result image to: {filename}")
        tensor = tensor.cpu().clone()
        tensor = tensor.squeeze(0)
        tensor = tensor * torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor + torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        tensor = tensor.clamp(0, 1)
        
        transform = transforms.ToPILImage()
        image = transform(tensor)
        image.save(filename)
        logger.info("Image saved successfully")
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        raise e

class StyleTransfer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = models.vgg19(weights='DEFAULT').features.to(self.device).eval()
        
        for param in self.model.parameters():
            param.requires_grad_(False)
    
    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1',
                     '5': 'conv2_1', 
                     '10': 'conv3_1', 
                     '19': 'conv4_1',
                     '21': 'conv4_2',
                     '28': 'conv5_1'}
        
        features = {}
        x = image
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                
        return features
    
    def gram_matrix(self, tensor):
        b, c, h, w = tensor.size()
        tensor = tensor.view(c, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram
    
    def style_transfer(self, content_image, style_image, num_steps=300):
        try:
            logger.info("Starting style transfer process")
            content_image = content_image.to(self.device)
            style_image = style_image.to(self.device)
            
            content_features = self.get_features(content_image)
            style_features = self.get_features(style_image)
            
            style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}
            
            target = content_image.clone().requires_grad_(True)
            
            style_weights = {'conv1_1': 1.,
                            'conv2_1': 0.75,
                            'conv3_1': 0.2,
                            'conv4_1': 0.2,
                            'conv5_1': 0.2}
            
            content_weight = 1
            style_weight = 1e6
            
            optimizer = torch.optim.Adam([target], lr=0.003)
            
            for step in range(num_steps):
                target_features = self.get_features(target)
                
                content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
                
                style_loss = 0
                for layer in style_weights:
                    target_feature = target_features[layer]
                    target_gram = self.gram_matrix(target_feature)
                    style_gram = style_grams[layer]
                    layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                    style_loss += layer_style_loss
                
                total_loss = content_weight * content_loss + style_weight * style_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if step % 20 == 0:  # Increased logging frequency
                    logger.info(f'Step {step}/{num_steps}: Style Loss: {style_loss.item():.4f}, Content Loss: {content_loss.item():.4f}')
            
            logger.info("Style transfer completed successfully")
            return target
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("GPU out of memory error occurred")
                raise RuntimeError("GPU out of memory. Try using a smaller image size.")
            logger.error(f"Runtime error during style transfer: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during style transfer: {str(e)}")
            raise e

@app.route('/')
def index():
    logger.info("Loading index page")
    return render_template('index.html')

@app.route('/transfer', methods=['POST'])
def transfer():
    logger.info("Starting new style transfer request")
    content_path = None
    style_path = None
    
    try:
        if 'content' not in request.files or 'style' not in request.files:
            logger.error("Missing files in request")
            return jsonify({'error': 'Missing files'}), 400
        
        content_file = request.files['content']
        style_file = request.files['style']
        
        if not (content_file and allowed_file(content_file.filename) and 
                style_file and allowed_file(style_file.filename)):
            logger.error("Invalid file type submitted")
            return jsonify({'error': 'Invalid file type. Please use PNG, JPG, JPEG, or GIF'}), 400
        
        logger.info(f"Processing content file: {content_file.filename}")
        logger.info(f"Processing style file: {style_file.filename}")
        
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_file.filename))
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_file.filename))
        
        content_file.save(content_path)
        style_file.save(style_path)
        
        # Load images
        content = load_image(content_path)
        style = load_image(style_path)
        
        # Perform style transfer
        style_transfer = StyleTransfer()
        result = style_transfer.style_transfer(content, style)
        
        # Save result
        result_filename = f'result_{int(time.time())}.jpg'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        save_image(result, result_path)
        
        # Clean up input files
        os.remove(content_path)
        os.remove(style_path)
        
        logger.info("Style transfer completed and result saved")
        return jsonify({'result': f'/static/uploads/{result_filename}'})
    
    except Exception as e:
        logger.error(f"Error during transfer process: {str(e)}")
        # Clean up files in case of error
        for path in [content_path, style_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up file: {path}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up file {path}: {str(cleanup_error)}")
        
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True) 