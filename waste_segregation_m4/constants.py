"""
Constants and configuration for the Waste Segregation System.
"""

# Waste categories
RECYCLABLES = ['plastic', 'metal', 'glass']
GENERAL_WASTE = ['cardboard', 'paper', 'trash', 'battery', 'clothes', 'organic', 'shoes']

# Default model configuration
DEFAULT_MODEL_KEY = "siglip2"  # Best accuracy from research
DEFAULT_CLASSIFICATION_COOLDOWN = 2.0  # seconds

# UI Configuration
DEFAULT_FPS = 10
MIN_FPS = 1
MAX_FPS = 30
RERUN_INTERVAL = 2.0  # seconds (controls flashing - higher = less flicker but less responsive)

# ROI Defaults
DEFAULT_ROI_X = 100
DEFAULT_ROI_Y = 50
DEFAULT_ROI_WIDTH = 800
DEFAULT_ROI_HEIGHT = 600
MAX_ROI_X = 1920
MAX_ROI_Y = 1080

# Model name for footer
MODEL_NAME = "prithivMLmods/Augmented-Waste-Classifier-SigLIP2"
