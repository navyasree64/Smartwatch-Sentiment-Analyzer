# Installation Guide

## Quick Fix for Installation Issues

If you encounter errors about missing C++ compilers when installing packages, try these solutions:

### Option 1: Install Pre-built Wheels (Recommended)

The updated `requirements.txt` now uses flexible version ranges that should work with pre-built wheels. Try installing again:

```bash
pip install -r requirements.txt
```

### Option 2: Install Visual Studio Build Tools (If Option 1 Fails)

If you still get compiler errors, install Visual Studio Build Tools:

1. Download and install **Microsoft C++ Build Tools** from:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. During installation, select "Desktop development with C++"

3. After installation, restart your terminal and try again:
   ```bash
   pip install -r requirements.txt
   ```

### Option 3: Use Pre-built Packages Only

If you want to avoid compilation entirely, install packages individually with `--only-binary`:

```bash
pip install --only-binary :all: flask pandas scikit-learn transformers torch numpy
```

### Option 4: Use Conda (Alternative)

If pip continues to have issues, consider using conda:

```bash
conda install flask pandas scikit-learn transformers pytorch numpy -c conda-forge
```

## Verify Installation

After installation, verify everything works:

```bash
python -c "import flask, pandas, sklearn, transformers, torch, numpy; print('All packages installed successfully!')"
```

## Running the Application

Once all packages are installed:

```bash
python app.py
```

The application will:
1. Train the classical ML model on your dataset
2. Load the transformer model (this may take a few minutes on first run)
3. Start the Flask server on http://localhost:5000

