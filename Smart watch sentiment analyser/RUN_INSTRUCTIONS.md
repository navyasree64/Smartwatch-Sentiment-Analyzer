# How to Run the Smartwatch Sentiment Analyzer

## Step 1: Install Python Dependencies

Open your terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

**If you encounter errors about missing C++ compilers**, try:

```bash
pip install --only-binary :all: -r requirements.txt
```

Or install packages individually:
```bash
pip install flask pandas scikit-learn transformers torch numpy nltk imbalanced-learn scipy
```

## Step 2: Run the Application

Simply run:

```bash
python app.py
```

## Step 3: Access the Web Interface

Once the server starts, you'll see output like:

```
Initializing models...
Training classical ML model...
============================================================
CLASSICAL ML MODEL TRAINING
============================================================
...
Starting Flask server...
 * Running on http://0.0.0.0:5000
```

Open your web browser and go to:

```
http://localhost:5000
```

or

```
http://127.0.0.1:5000
```

## What Happens When You Run

1. **Model Training** (First Time):
   - Loads the dataset from `Dataset/7817_1.csv`
   - Trains a Naive Bayes classical ML model
   - Applies SMOTE for data balancing
   - Loads the Transformer model
   - This may take 2-5 minutes on first run

2. **Server Starts**:
   - Flask server runs on port 5000
   - Web interface is ready to use

## Using the Application

1. Enter a smartwatch review in the text area
2. Click "Analyze Sentiment"
3. View results from:
   - Classical ML model (Naive Bayes)
   - Transformer model
   - Sentence-level analysis (Transformer only)

## Troubleshooting

### Port Already in Use
If port 5000 is busy, edit `app.py` and change:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```
to a different port like `port=5001`

### Missing Dataset
Make sure `Dataset/7817_1.csv` exists in the project folder

### Model Download Issues
The transformer model will download automatically on first run (may take a few minutes)

### Memory Issues
If you get memory errors, reduce features in `app.py`:
- Change `max_features=20000` to `max_features=10000`

## Quick Start (All Commands)

```bash
# Navigate to project directory
cd "C:\Users\madal\Desktop\Smart watch sentiment analyser"

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Open browser to http://localhost:5000
```

