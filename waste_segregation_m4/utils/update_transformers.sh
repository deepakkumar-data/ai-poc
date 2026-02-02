#!/bin/bash

# Script to update transformers to support mobilenet_v3 architecture

echo "🔄 Updating transformers to support mobilenet_v3 architecture..."

# Check current version
echo "Current transformers version:"
python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "transformers not installed"

echo ""
echo "Updating transformers..."

# Try upgrading first
pip install --upgrade transformers

# Check if upgrade worked
python -c "from transformers import AutoConfig; config = AutoConfig.from_pretrained('AmadFR/ecovision_mobilenetv3', trust_remote_code=True); print('✅ Config loaded successfully')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Standard upgrade didn't work. Installing from source..."
    pip install git+https://github.com/huggingface/transformers.git
fi

echo ""
echo "✅ Transformers update complete!"
echo "   Restart your Python session and try loading the model again."
