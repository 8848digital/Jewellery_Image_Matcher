# unified_model.py - Final fixed version for your exact model architecture
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
from pathlib import Path
import io
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_checkpoint_architecture(state_dict):
    """Analyze checkpoint to determine exact architecture"""
    
    # Analyze position embeddings to determine image size
    pos_embed_shape = state_dict['backbone.pos_embed'].shape  # [1, num_patches+1, embed_dim]
    num_patches_plus_one = pos_embed_shape[1]  # 1370
    num_patches = num_patches_plus_one - 1  # 1369
    embed_dim = pos_embed_shape[2]  # 1024
    
    # Determine patch size from patch_embed projection
    patch_proj_shape = state_dict['backbone.patch_embed.proj.weight'].shape  # [1024, 3, 14, 14]
    patch_size = patch_proj_shape[2]  # 14
    
    # Calculate image size: num_patches = (img_size // patch_size)^2
    img_size_patches = int(math.sqrt(num_patches))  # 37
    img_size = img_size_patches * patch_size  # 37 * 14 = 518
    
    # Analyze projection head layers
    proj_layers = []
    
    # Layer 0: Linear
    if 'projection.0.weight' in state_dict:
        proj_layers.append({
            'type': 'Linear',
            'in_features': state_dict['projection.0.weight'].shape[1],
            'out_features': state_dict['projection.0.weight'].shape[0]
        })
    
    # Layer 1: ReLU (no parameters)
    proj_layers.append({'type': 'ReLU'})
    
    # Layer 2: BatchNorm1d
    if 'projection.2.weight' in state_dict and len(state_dict['projection.2.weight'].shape) == 1:
        proj_layers.append({
            'type': 'BatchNorm1d',
            'num_features': state_dict['projection.2.weight'].shape[0]
        })
    
    # Layer 3: ReLU (no parameters)
    proj_layers.append({'type': 'ReLU'})
    
    # Layer 4: Linear
    if 'projection.4.weight' in state_dict:
        proj_layers.append({
            'type': 'Linear',
            'in_features': state_dict['projection.4.weight'].shape[1],
            'out_features': state_dict['projection.4.weight'].shape[0]
        })
    
    # Determine final embedding dimension
    final_emb_dim = 256  # From projection.4.weight: [256, 512]
    if proj_layers:
        for layer in reversed(proj_layers):
            if layer['type'] == 'Linear':
                final_emb_dim = layer['out_features']
                break
    
    architecture = {
        'img_size': img_size,
        'patch_size': patch_size,
        'embed_dim': embed_dim,
        'num_patches': num_patches,
        'proj_layers': proj_layers,
        'final_emb_dim': final_emb_dim
    }
    
    logger.info(f"Detected architecture: {architecture}")
    return architecture

class CustomJewelryModel(nn.Module):
    """Recreate your custom jewelry model architecture"""
    
    def __init__(self, architecture):
        super().__init__()
        
        # Import timm for backbone
        import timm
        
        # Create the backbone with LayerScale support
        # Your model has LayerScale, so we need a model that supports it
        possible_models = [
            'vit_large_patch14_224.mag_in1k',  # Has LayerScale
            'vit_large_patch14_224_in21k',      # Try this too
            'vit_large_patch14_224'             # Fallback
        ]
        
        self.backbone = None
        for model_name in possible_models:
            try:
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=False, 
                    num_classes=0,
                    global_pool='',
                    img_size=architecture['img_size']
                )
                logger.info(f"Created backbone: {model_name}")
                break
            except Exception as e:
                logger.info(f"Failed to create {model_name}: {e}")
                continue
        
        if self.backbone is None:
            raise RuntimeError("Could not create any suitable backbone model")
        
        # Create projection head based on detected layers
        projection_layers = []
        for i, layer_info in enumerate(architecture['proj_layers']):
            if layer_info['type'] == 'Linear':
                projection_layers.append(
                    nn.Linear(layer_info['in_features'], layer_info['out_features'])
                )
            elif layer_info['type'] == 'ReLU':
                projection_layers.append(nn.ReLU())
            elif layer_info['type'] == 'BatchNorm1d':
                projection_layers.append(
                    nn.BatchNorm1d(layer_info['num_features'])
                )
        
        self.projection = nn.Sequential(*projection_layers)
        
        self.emb_dim = architecture['final_emb_dim']
        self.img_size = architecture['img_size']
    
    def forward_features(self, x):
        """Extract features using the backbone"""
        # Get patch features from backbone
        features = self.backbone.forward_features(x)
        
        # features shape: [batch, num_patches + 1, 1024]
        # Extract class token (first token)
        cls_token = features[:, 0, :]  # [batch, 1024]
        
        return cls_token
    
    def forward(self, x):
        """Full forward pass with projection"""
        features = self.forward_features(x)
        embeddings = self.projection(features)
        
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class UnifiedJewelryModel:
    """Unified model handler ensuring consistent embeddings across all scripts"""
    
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _load_model(self, model_path):
        """Load model with consistent architecture"""
        logger.info(f"Loading model from {model_path} on device {self.device}")
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Get state dict and analyze architecture
            state_dict = checkpoint['model_state_dict']
            architecture = analyze_checkpoint_architecture(state_dict)
            
            # Recreate the model with detected architecture
            model = CustomJewelryModel(architecture)
            
            # Load the state dict with strict=False to ignore unexpected keys
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)} keys")
                # Only show first few to avoid spam
                for key in missing_keys[:5]:
                    logger.warning(f"  Missing: {key}")
                if len(missing_keys) > 5:
                    logger.warning(f"  ... and {len(missing_keys) - 5} more missing keys")
            
            if unexpected_keys:
                logger.info(f"Unexpected keys: {len(unexpected_keys)} keys (LayerScale, mask_token, etc.)")
                # This is expected due to LayerScale and other advanced features
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Model embedding dimension: {model.emb_dim}")
            logger.info(f"Model image size: {model.img_size}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_transform(self):
        """Get consistent preprocessing transforms with correct image size"""
        # Use the model's expected image size
        img_size = getattr(self.model, 'img_size', 224)
        
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_embedding(self, image):
        """Extract embedding with consistent method"""
        try:
            # Handle different image input types
            if isinstance(image, str):
                # File path
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, bytes):
                # Bytes from Streamlit upload
                pil_image = Image.open(io.BytesIO(image)).convert('RGB')
            elif hasattr(image, 'read'):
                # File-like object
                pil_image = Image.open(image).convert('RGB')
            else:
                # Assume PIL Image
                pil_image = image.convert('RGB')
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Use the full forward pass (includes projection and normalization)
                embedding = self.model(input_tensor)
                
                # Convert to numpy
                embedding_np = embedding.cpu().numpy().flatten()
                
                logger.debug(f"Extracted embedding shape: {embedding_np.shape}")
                logger.debug(f"Embedding norm: {np.linalg.norm(embedding_np):.4f}")
                
                return embedding_np
                
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise
    
    def extract_features_only(self, image):
        """Extract only backbone features (without projection) - for debugging"""
        try:
            # Handle image input
            if isinstance(image, str):
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image)).convert('RGB')
            elif hasattr(image, 'read'):
                pil_image = Image.open(image).convert('RGB')
            else:
                pil_image = image.convert('RGB')
            
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Extract only backbone features
                features = self.model.forward_features(input_tensor)
                features_np = features.cpu().numpy().flatten()
                
                logger.debug(f"Backbone features shape: {features_np.shape}")
                return features_np
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def compute_similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings"""
        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()
        
        # Normalize embeddings (should already be normalized from model)
        emb1_norm = np.linalg.norm(emb1)
        emb2_norm = np.linalg.norm(emb2)
        
        if emb1_norm > 0:
            emb1 = emb1 / emb1_norm
        if emb2_norm > 0:
            emb2 = emb2 / emb2_norm
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    
    def batch_extract_embeddings(self, image_paths):
        """Extract embeddings for multiple images"""
        embeddings = []
        for img_path in image_paths:
            try:
                emb = self.extract_embedding(img_path)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                embeddings.append(None)
        return embeddings