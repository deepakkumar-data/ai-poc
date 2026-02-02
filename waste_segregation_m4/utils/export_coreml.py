"""
Core ML Model Export Script
Converts PyTorch models to Core ML format for Neural Engine acceleration
"""

import torch
import numpy as np
import coremltools as ct
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
from pathlib import Path


def export_to_coreml(
    model_name: str = "AmadFR/ecovision_mobilenetv3",
    output_dir: str = "../models",
    output_name: str = "ecovision_mobilenetv3.mlpackage"
):
    """
    Export Hugging Face model to Core ML format.
    
    Args:
        model_name: Hugging Face model identifier
        output_dir: Directory to save the Core ML model
        output_name: Name of the output Core ML model file
    """
    print(f"📦 Loading model: {model_name}")
    
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    
    # Get input shape from processor
    # Typically 224x224 for MobileNet models
    image_size = processor.size.get('shortest_edge', 224)
    
    print(f"📐 Input image size: {image_size}x{image_size}")
    
    # Create example input
    example_input = torch.randn(1, 3, image_size, image_size)
    
    # Trace the model
    print("🔄 Tracing PyTorch model...")
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to Core ML
    print("🔄 Converting to Core ML format...")
    
    # Define input specification
    input_shape = ct.Shape(shape=(1, 3, image_size, image_size))
    scale = 1.0 / 255.0  # Normalization scale
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="pixel_values",
                shape=input_shape,
                dtype=np.float32
            )
        ],
        outputs=[ct.TensorType(name="logits")],
        compute_units=ct.ComputeUnit.ALL,  # Use Neural Engine + CPU + GPU
        minimum_deployment_target=ct.target.macOS13,  # macOS 13+ for Neural Engine
    )
    
    # Add metadata
    mlmodel.author = "Waste Segregation System"
    mlmodel.short_description = "Waste classification model optimized for Apple Neural Engine"
    mlmodel.version = "1.0"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    output_file = output_path / output_name
    mlmodel.save(str(output_file))
    
    print(f"✅ Core ML model saved to: {output_file}")
    print(f"📊 Model size: {output_file.stat().st_size / (1024 * 1024):.2f} MB")
    
    return str(output_file)


def verify_coreml_model(model_path: str):
    """Verify that the Core ML model can be loaded and used."""
    import coremltools as ct
    
    print(f"🔍 Verifying Core ML model: {model_path}")
    
    try:
        # Load model
        mlmodel = ct.models.MLModel(model_path)
        
        # Print model details
        print("✅ Model loaded successfully")
        print(f"   Input: {mlmodel.input_description}")
        print(f"   Output: {mlmodel.output_description}")
        print(f"   Compute Units: {mlmodel.compute_unit}")
        
        # Test prediction (if possible)
        # Note: This requires a sample image
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying model: {e}")
        return False


if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Export PyTorch model to Core ML")
    parser.add_argument(
        "--model",
        type=str,
        default="AmadFR/ecovision_mobilenetv3",
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../models",
        help="Output directory for Core ML model"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="ecovision_mobilenetv3.mlpackage",
        help="Output filename"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported model"
    )
    
    args = parser.parse_args()
    
    # Export model
    model_path = export_to_coreml(
        model_name=args.model,
        output_dir=args.output_dir,
        output_name=args.output_name
    )
    
    # Verify if requested
    if args.verify:
        verify_coreml_model(model_path)
    
    print("\n✨ Export complete!")
    print("   Use the Core ML model with:")
    print("   - Core ML framework for Neural Engine acceleration")
    print("   - MLX for additional optimizations")
