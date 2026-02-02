#!/usr/bin/env python3
"""
MPS (Metal Performance Shaders) and Neural Engine Verification Script
for Mac Mini M4 with 16-core Neural Engine
"""

import sys
import platform

def check_system_info():
    """Display system information"""
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print()

def check_pytorch():
    """Check PyTorch installation and MPS availability"""
    print("=" * 60)
    print("PyTorch MPS Backend Check")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check if MPS is available
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) backend is available")
            
            # Check if MPS is built (compiled with MPS support)
            if torch.backends.mps.is_built():
                print("✅ MPS backend is built and ready to use")
                
                # Create a test tensor on MPS
                try:
                    device = torch.device("mps")
                    x = torch.randn(3, 3).to(device)
                    y = torch.randn(3, 3).to(device)
                    z = x @ y  # Matrix multiplication test
                    print("✅ MPS device test passed - tensor operations working")
                    print(f"   Device: {device}")
                    
                    # Get device properties if available
                    if hasattr(torch.backends.mps, 'is_available'):
                        print(f"   MPS available: {torch.backends.mps.is_available()}")
                    
                except Exception as e:
                    print(f"⚠️  MPS device test failed: {e}")
            else:
                print("❌ MPS backend is not built (PyTorch compiled without MPS support)")
        else:
            print("❌ MPS backend is not available")
            print("   This may indicate:")
            print("   - macOS version < 12.3 (Monterey)")
            print("   - PyTorch not compiled with MPS support")
            print("   - Running on non-Apple Silicon hardware")
        
        # Display available devices
        print("\nAvailable PyTorch devices:")
        if torch.cuda.is_available():
            print(f"  - CUDA: {torch.cuda.get_device_name(0)}")
        if torch.backends.mps.is_available():
            print("  - MPS: Metal Performance Shaders (Apple Silicon)")
        print("  - CPU: Always available")
        
    except ImportError:
        print("❌ PyTorch is not installed")
        return False
    
    print()
    return True

def check_neural_engine():
    """Check Neural Engine accessibility via Core ML"""
    print("=" * 60)
    print("Neural Engine Check (16-core on M4)")
    print("=" * 60)
    
    try:
        import coremltools as ct
        print("✅ Core ML is available")
        print("   Neural Engine can be accessed via Core ML framework")
    except ImportError:
        print("⚠️  Core ML tools not installed (optional)")
        print("   Install with: pip install coremltools")
    
    # Check MLX (Apple's ML framework that uses Neural Engine)
    try:
        import mlx.core as mx
        print("✅ MLX is available")
        print("   MLX can leverage the Neural Engine for ML operations")
        
        # Test MLX with a simple operation
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([4.0, 5.0, 6.0])
        z = x * y
        print(f"   MLX test operation successful: {z}")
        
    except ImportError:
        print("⚠️  MLX is not installed")
        print("   Install with: pip install mlx")
    except Exception as e:
        print(f"⚠️  MLX test failed: {e}")
    
    print()
    
    # System-level check using system_profiler
    import subprocess
    try:
        result = subprocess.run(
            ['system_profiler', 'SPHardwareDataType'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if 'Neural Engine' in result.stdout:
            print("System Hardware Info:")
            for line in result.stdout.split('\n'):
                if 'Neural Engine' in line or 'Chip' in line:
                    print(f"  {line.strip()}")
    except Exception as e:
        print(f"⚠️  Could not query system hardware: {e}")
    
    print()

def check_opencv():
    """Check OpenCV installation"""
    print("=" * 60)
    print("OpenCV Check")
    print("=" * 60)
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        print(f"   Build info: {cv2.getBuildInformation().split(chr(10))[0:3]}")
    except ImportError:
        print("❌ OpenCV is not installed")
    
    print()

def check_transformers():
    """Check Transformers library"""
    print("=" * 60)
    print("Transformers Check")
    print("=" * 60)
    
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers is not installed")
    
    print()

def main():
    """Main verification function"""
    print("\n🔍 Verifying MPS and Neural Engine Setup for Mac Mini M4\n")
    
    check_system_info()
    pytorch_ok = check_pytorch()
    check_neural_engine()
    check_opencv()
    check_transformers()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    if pytorch_ok:
        import torch
        if torch.backends.mps.is_available():
            print("✅ Your system is ready for GPU-accelerated ML on Apple Silicon!")
            print("   Use device='mps' when creating tensors in PyTorch")
        else:
            print("⚠️  MPS is not available. Check your PyTorch installation.")
    else:
        print("❌ PyTorch is not properly installed")
    
    print()

if __name__ == "__main__":
    main()
