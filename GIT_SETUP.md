# Git Repository Setup Guide

This guide will help you prepare and commit this repository to git.

## Pre-Commit Checklist

✅ **Files to Commit:**
- All Python source files (`.py`)
- Configuration files (`requirements.txt`, `constants.py`)
- Documentation (`README.md` files)
- Setup scripts (`setup.sh`, `setup_conda.sh`)
- Utility scripts in `utils/`
- `.gitignore` files
- `.gitkeep` files (to preserve directory structure)

❌ **Files Ignored (via .gitignore):**
- Virtual environments (`venv/`, `.venv/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Model files (`*.pth`, `*.pt`, `*.onnx`, `*.mlmodel`)
- Test videos (`*.mp4`, `*.avi`, etc.)
- IDE files (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)
- Hugging Face cache
- Streamlit cache

## Initial Git Setup

If this is a new repository:

```bash
# Navigate to project root
cd /Users/deepak/Documents/Ambia/Aria/testdb

# Initialize git repository (if not already initialized)
git init

# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status

# Create initial commit
git commit -m "Initial commit: Waste Segregation AI System for Mac Mini M4"


- Complete waste classification system with MPS acceleration
- Streamlit UI for real-time and video processing
- Conveyor belt object detection and tracking
- Motion detection and trigger line classification
- Comprehensive documentation and utilities"
```

## Recommended Git Workflow

### 1. Check Repository Status

```bash
git status
```

This shows:
- Files that will be committed (staged)
- Files that are modified but not staged
- Files that are untracked
- Files that are ignored

### 2. Review What Will Be Committed

```bash
# See detailed changes
git diff

# See staged changes
git diff --staged
```

### 3. Stage Files

```bash
# Stage all changes
git add .

# Or stage specific files
git add waste_segregation_m4/app.py
git add waste_segregation_m4/README.md
```

### 4. Commit Changes

```bash
git commit -m "Descriptive commit message"
```

### 5. Add Remote Repository (if pushing to GitHub/GitLab)

```bash
# Add remote origin
git remote add origin <repository-url>

# Verify remote
git remote -v

# Push to remote
git push -u origin main
# or
git push -u origin master
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

**Good examples:**
- `"Add motion detection to conveyor engine"`
- `"Fix duplicate counting in session statistics"`
- `"Update README with test video instructions"`
- `"Improve Streamlit UI for operator experience"`

**Bad examples:**
- `"fix"`
- `"update"`
- `"changes"`

## Branch Strategy (Optional)

For larger projects, consider using branches:

```bash
# Create a feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Switch back to main
git checkout main

# Merge feature branch
git merge feature/new-feature
```

## Files Structure in Git

```
testdb/
├── .gitignore                 # Root gitignore
├── README.md                  # Root README
├── setup.sh                   # Setup script
├── setup_conda.sh            # Conda setup script
└── waste_segregation_m4/
    ├── .gitignore            # Project-specific gitignore
    ├── README.md             # Detailed project README
    ├── app.py                # Main Streamlit app
    ├── classifier.py         # Model classifier
    ├── conveyor_engine.py   # Video processing engine
    ├── constants.py         # Configuration constants
    ├── requirements.txt     # Python dependencies
    ├── models/
    │   └── .gitkeep         # Preserve directory structure
    ├── test_videos/
    │   └── .gitkeep         # Preserve directory structure
    └── utils/
        ├── check_mps.py
        ├── check_model.py
        ├── download_test_video.py
        ├── export_coreml.py
        ├── fix_camera_permissions.sh
        ├── get_test_video.sh
        ├── quick_camera_test.py
        ├── test_camera.py
        └── update_transformers.sh
```

## Important Notes

1. **Model Files**: Large model files (`.pth`, `.pt`, `.onnx`) are excluded. Users will download models automatically on first run via Hugging Face.

2. **Virtual Environment**: The `venv/` directory is excluded. Users should create their own virtual environment using the setup scripts.

3. **Test Videos**: Video files are excluded. Users can download test videos using the provided utilities.

4. **Cache Files**: All cache directories (Hugging Face, Streamlit, Python) are excluded.

## Verification

After committing, verify the repository:

```bash
# Check repository size (should be reasonable, not GBs)
du -sh .

# Verify important files are tracked
git ls-files | grep -E "(app.py|classifier.py|README.md)"

# Verify ignored files are not tracked
git ls-files | grep -E "(venv|__pycache__|\.pth|\.mp4)" || echo "Good: Ignored files not tracked"
```

## Troubleshooting

### If you accidentally committed large files:

```bash
# Remove from git history (use with caution)
git rm --cached large_file.pth
git commit -m "Remove large file from tracking"
```

### If .gitignore isn't working:

```bash
# Remove cached files
git rm -r --cached .
git add .
git commit -m "Update .gitignore"
```

### If you need to exclude a file that's already tracked:

```bash
# Remove from tracking but keep locally
git rm --cached file_to_ignore
# Add to .gitignore
echo "file_to_ignore" >> .gitignore
git add .gitignore
git commit -m "Ignore file_to_ignore"
```
