#!/usr/bin/env python3
"""
Check if a Hugging Face model exists and what files it contains
"""

import sys
from huggingface_hub import HfApi, list_repo_files
from transformers import AutoConfig

def check_model(model_name: str):
    """Check if model exists and what files it has."""
    print(f"🔍 Checking model: {model_name}")
    print("=" * 60)
    
    api = HfApi()
    
    try:
        # Check if model exists
        model_info = api.model_info(model_name)
        print(f"✅ Model exists on Hugging Face")
        print(f"   ID: {model_info.id}")
        print(f"   Author: {model_info.author}")
        
        # List files in the repository
        print("\n📁 Files in repository:")
        files = list_repo_files(model_name, repo_type="model")
        for file in files:
            print(f"   - {file}")
        
        # Check for custom code files
        has_custom_code = any(
            f.endswith('.py') and ('modeling' in f or 'configuration' in f)
            for f in files
        )
        
        print(f"\n🔧 Custom code files: {'✅ Found' if has_custom_code else '❌ Not found'}")
        
        # Try to load config
        print("\n📋 Trying to load config...")
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            print(f"   ✅ Config loaded successfully")
            print(f"   Model type: {config.model_type if hasattr(config, 'model_type') else 'Unknown'}")
        except Exception as e:
            print(f"   ❌ Failed to load config: {e}")
            print(f"   This suggests the model architecture is not supported.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking model: {e}")
        return False

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "AmadFR/ecovision_mobilenetv3"
    check_model(model_name)
