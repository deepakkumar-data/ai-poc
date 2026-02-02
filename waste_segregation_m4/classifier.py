"""
Waste Classification Module
Handles local inference using Hugging Face waste classification models
Optimized for Mac Mini M4 with MPS acceleration

Based on research: Technical Evaluation and Implementation Strategy for 
Localized AI-Driven Waste Segregation Systems Using Hugging Face Architectures

Supported Models (from research):
- prithivMLmods/Augmented-Waste-Classifier-SigLIP2: SigLIP2-Base, ~400M params, 6 classes
- prithivMLmods/Trash-Net: SigLIP2-Base, ~400M params, 6 classes
- watersplash/waste-classification: ViT-Base, 86M params, 12 classes
- randyver/trash-classification-cnn: Custom CNN, <10M params, 6 classes
- kendrickfff/my_resnet50_garbage_classification: ResNet50, 12 categories
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from typing import Tuple, Optional, Union
import warnings

# Enable MPS fallback for unsupported operators
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


class WasteClassifier:
    """
    Waste classification using Hugging Face models.
    Optimized for waste categories with MPS acceleration on Apple Silicon.
    
    Recommended models (from research):
    - prithivMLmods/Augmented-Waste-Classifier-SigLIP2: Best accuracy, 6 classes
    - watersplash/waste-classification: ViT-Base, 12 classes, good balance
    - randyver/trash-classification-cnn: Lightweight, 6 classes, edge devices
    """
    
    # Recommended models from research (ordered by performance/complexity)
    RECOMMENDED_MODELS = {
        "siglip2": "prithivMLmods/Augmented-Waste-Classifier-SigLIP2",  # Best accuracy, 6 classes
        "trashnet": "prithivMLmods/Trash-Net",  # SigLIP2, 6 classes
        "vit_waste": "watersplash/waste-classification",  # ViT-Base, 12 classes
        "resnet50": "kendrickfff/my_resnet50_garbage_classification",  # ResNet50, 12 classes
        "cnn_lightweight": "randyver/trash-classification-cnn",  # Lightweight CNN, 6 classes
    }
    
    # Common waste categories (varies by model)
    COMMON_CATEGORIES = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash",
        "battery",
        "clothes",
        "organic",
        "shoes"
    ]
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        force_mps: bool = True,
        use_onnx: bool = False,
        model_key: Optional[str] = None
    ):
        """
        Initialize the Waste Classifier.
        
        Args:
            model_name: Hugging Face model identifier. If None, uses recommended model.
            device: Target device ('mps', 'cpu', 'cuda'). If None, auto-detects.
            force_mps: If True, forces MPS device on Apple Silicon (default: True)
            use_onnx: If True, uses ONNX Runtime for optimized inference (default: False)
            model_key: Short key for recommended model ('siglip2', 'trashnet', 'vit_waste', etc.)
        """
        # Determine model name
        if model_key and model_key in self.RECOMMENDED_MODELS:
            self.model_name = self.RECOMMENDED_MODELS[model_key]
            print(f"📋 Using recommended model: {model_key} -> {self.model_name}")
        elif model_name:
            self.model_name = model_name
        else:
            # Default to best performing model from research
            self.model_name = self.RECOMMENDED_MODELS["siglip2"]
            print(f"📋 Using default recommended model: {self.model_name}")
        
        self.use_onnx = use_onnx
        self.device = self._setup_device(device, force_mps)
        self.model = None
        self.processor = None
        self._load_model()
    
    def _setup_device(self, device: Optional[str], force_mps: bool) -> torch.device:
        """
        Setup and configure the PyTorch device.
        Cross-platform: MPS (macOS), CUDA (Windows/Linux), CPU (fallback)
        
        Args:
            device: Preferred device
            force_mps: Whether to force MPS on Apple Silicon (macOS only)
            
        Returns:
            Configured torch.device
        """
        if device is not None:
            return torch.device(device)
        
        # Check for MPS (macOS Apple Silicon)
        if force_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                print(f"✅ Using MPS device for GPU acceleration on Apple Silicon")
                return torch.device("mps")
            else:
                print("⚠️  MPS not built, falling back to CPU")
                return torch.device("cpu")
        # Check for CUDA (Windows/Linux with NVIDIA GPU)
        elif torch.cuda.is_available():
            print("✅ Using CUDA device for GPU acceleration")
            return torch.device("cuda")
        else:
            print("ℹ️  Using CPU device")
            return torch.device("cpu")
    
    def _load_model(self):
        """Load the model and processor from Hugging Face."""
        try:
            print(f"📦 Loading model: {self.model_name}")
            print("   This may take a moment on first download...")
            
            # Check transformers version
            import transformers
            transformers_version = transformers.__version__
            print(f"   Using transformers version: {transformers_version}")
            
            # Try ONNX Runtime if requested (faster inference)
            if self.use_onnx:
                try:
                    from optimum.onnxruntime import ORTModelForImageClassification
                    print("   🔄 Attempting to load ONNX optimized model...")
                    self.model = ORTModelForImageClassification.from_pretrained(
                        self.model_name,
                        export=True  # Export to ONNX if not already exported
                    )
                    self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                    print("   ✅ ONNX model loaded successfully")
                    return
                except ImportError:
                    print("   ⚠️  ONNX Runtime not available, falling back to PyTorch")
                    print("   Install with: pip install optimum[onnxruntime]")
                except Exception as onnx_error:
                    print(f"   ⚠️  ONNX loading failed: {onnx_error}")
                    print("   Falling back to PyTorch model")
            
            # Load config with trust_remote_code for custom architectures
            try:
                config = AutoConfig.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                print("   ✅ Config loaded with trust_remote_code")
            except Exception as config_error:
                error_msg = str(config_error)
                if "does not recognize this architecture" in error_msg:
                    # Try without trust_remote_code for standard architectures
                    config = AutoConfig.from_pretrained(self.model_name)
                    print("   ✅ Config loaded (standard architecture)")
                else:
                    raise
            
            # Load the image processor
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            except (ValueError, KeyError, TypeError):
                # Fallback: try without trust_remote_code for standard processors
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # Load the model with trust_remote_code for custom architectures
            try:
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Use float32 for MPS compatibility
                    trust_remote_code=True
                )
            except Exception:
                # Fallback: try without trust_remote_code for standard architectures
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
            
            # Move model to device (skip for ONNX models)
            if not self.use_onnx or not hasattr(self.model, 'model'):
                self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print(f"✅ Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            error_msg = str(e)
            # Suggest using recommended models
            recommended = "\n".join([f"   - {k}: {v}" for k, v in self.RECOMMENDED_MODELS.items()])
            raise RuntimeError(
                f"❌ Failed to load model '{self.model_name}': {error_msg}\n\n"
                "SOLUTION: Use a recommended model from research:\n"
                f"{recommended}\n\n"
                "Example:\n"
                "  classifier = WasteClassifier(model_key='siglip2')\n"
                "  or\n"
                "  classifier = WasteClassifier(model_name='prithivMLmods/Augmented-Waste-Classifier-SigLIP2')\n\n"
                "See MODEL_RESEARCH.md for model comparison and selection guide."
            )
    
    def _preprocess_image(
        self, 
        image: Union[np.ndarray, Image.Image, str]
    ) -> torch.Tensor:
        """
        Preprocess image for model inference.
        Handles BGR-to-RGB conversion for OpenCV images and normalization.
        
        Args:
            image: Input image (OpenCV BGR array, PIL Image, or file path)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image if path is provided
        if isinstance(image, str):
            # Try OpenCV first (BGR)
            try:
                image = cv2.imread(image)
                if image is None:
                    raise ValueError(f"Could not load image from path: {image}")
            except Exception:
                # Fallback to PIL
                image = Image.open(image).convert('RGB')
        
        # Convert OpenCV BGR to RGB if needed
        if isinstance(image, np.ndarray):
            # Check if it's a BGR image (OpenCV format)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Convert numpy array to PIL Image
            image = Image.fromarray(image)
        
        # Use AutoImageProcessor for normalization and tensor conversion
        # This handles the proper preprocessing for the model
        inputs = self.processor(image, return_tensors="pt")
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def predict(
        self, 
        image: Union[np.ndarray, Image.Image, str],
        top_k: int = 1
    ) -> Tuple[str, float]:
        """
        Predict waste category from image.
        
        Args:
            image: Input image (OpenCV BGR array, PIL Image, or file path)
            top_k: Number of top predictions to return (default: 1)
            
        Returns:
            Tuple of (label, confidence_score) for top-1 prediction
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Preprocess image
        inputs = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            try:
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)
                
                # Get top-1 result
                top_prob = top_probs[0][0].item()
                top_idx = top_indices[0][0].item()
                
                # Get label from model's id2label mapping
                label = self.model.config.id2label.get(top_idx, f"category_{top_idx}")
                
                return label, top_prob
                
            except RuntimeError as e:
                # Handle MPS-specific errors with fallback
                if "mps" in str(e).lower() or "metal" in str(e).lower():
                    if self.device.type == "mps":
                        print("⚠️  MPS operation failed, falling back to CPU for this inference")
                        # Temporarily move to CPU
                        inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
                        model_cpu = self.model.cpu()
                        model_cpu.eval()
                        
                        with torch.no_grad():
                            outputs = model_cpu(**inputs_cpu)
                            logits = outputs.logits
                            probabilities = torch.nn.functional.softmax(logits, dim=-1)
                            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)
                            top_prob = top_probs[0][0].item()
                            top_idx = top_indices[0][0].item()
                            label = model_cpu.config.id2label.get(top_idx, f"category_{top_idx}")
                        
                        # Move model back to MPS
                        self.model = self.model.to(self.device)
                        return label, top_prob
                raise
    
    def predict_top_k(
        self, 
        image: Union[np.ndarray, Image.Image, str],
        top_k: int = 5
    ) -> list:
        """
        Get top-k predictions with labels and confidence scores.
        
        Args:
            image: Input image (OpenCV BGR array, PIL Image, or file path)
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (label, confidence_score) sorted by confidence
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Preprocess image
        inputs = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            try:
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)
                
                # Format results
                results = []
                for i in range(top_k):
                    prob = top_probs[0][i].item()
                    idx = top_indices[0][i].item()
                    label = self.model.config.id2label.get(idx, f"category_{idx}")
                    results.append((label, prob))
                
                return results
                
            except RuntimeError as e:
                # Handle MPS-specific errors with fallback
                if "mps" in str(e).lower() or "metal" in str(e).lower():
                    if self.device.type == "mps":
                        print("⚠️  MPS operation failed, falling back to CPU for this inference")
                        inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
                        model_cpu = self.model.cpu()
                        model_cpu.eval()
                        
                        with torch.no_grad():
                            outputs = model_cpu(**inputs_cpu)
                            logits = outputs.logits
                            probabilities = torch.nn.functional.softmax(logits, dim=-1)
                            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)
                            
                            results = []
                            for i in range(top_k):
                                prob = top_probs[0][i].item()
                                idx = top_indices[0][i].item()
                                label = model_cpu.config.id2label.get(idx, f"category_{idx}")
                                results.append((label, prob))
                        
                        self.model = self.model.to(self.device)
                        return results
                raise
    
    def get_device(self) -> str:
        """Get the current device being used."""
        return str(self.device)
    
    def get_categories(self) -> list:
        """Get the list of waste categories from the loaded model."""
        if self.model and hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
            # Return categories from the actual model
            categories = list(self.model.config.id2label.values())
            return categories
        # Fallback to common categories
        return self.COMMON_CATEGORIES.copy()
    
    def get_recommended_models(self) -> dict:
        """Get dictionary of recommended models from research."""
        return self.RECOMMENDED_MODELS.copy()


# Example usage
if __name__ == "__main__":
    print("🔍 Available recommended models:")
    temp_classifier = WasteClassifier.__new__(WasteClassifier)  # Create instance without init
    for key, model_name in temp_classifier.RECOMMENDED_MODELS.items():
        print(f"   {key}: {model_name}")
    
    print("\n📦 Initializing classifier with recommended model...")
    # Initialize classifier with best performing model from research
    classifier = WasteClassifier(model_key="siglip2", force_mps=True)
    
    # Example: Predict from image path
    # label, confidence = classifier.predict("path/to/image.jpg")
    # print(f"Predicted: {label} (confidence: {confidence:.2%})")
    
    print(f"✅ WasteClassifier initialized on device: {classifier.get_device()}")
    print(f"📋 Supported categories: {', '.join(classifier.get_categories())}")
