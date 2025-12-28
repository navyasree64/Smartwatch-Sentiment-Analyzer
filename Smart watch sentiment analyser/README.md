# Smartwatch Sentiment Analyzer

A web application that analyzes sentiment of smartwatch reviews using two different approaches:
1. **Classical ML Model**: Naive Bayes classifier with TF-IDF vectorization
2. **Transformer Model**: RoBERTa-based sentiment analysis model

## Features

- Real-time sentiment analysis of review text
- Side-by-side comparison of Classical ML vs Transformer models
- Model performance comparison on dataset
- Modern, responsive web interface

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the dataset file at `Dataset/7817_1.csv`

## Running the Application

### Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
python app.py
```

3. **Open your browser:**
```
http://localhost:5000
```

### First Run

On the first run, the application will:
- Load and preprocess the dataset
- Balance the data using SMOTE
- Train a Naive Bayes classical ML model
- Apply overfitting/underfitting prevention techniques
- Download the Transformer model (may take a few minutes)
- Start the Flask server

**Note:** Initial training may take 2-5 minutes. You'll see detailed progress in the console.

### What You'll See

- **Training Output:** Detailed metrics for the Naive Bayes model
- **Web Interface:** Clean UI showing all model predictions
- **Sentence Analysis:** Transformer model provides sentence-level sentiment

### Troubleshooting

- **Port in use?** Change port in `app.py` (line 248)
- **Missing dependencies?** Run `pip install -r requirements.txt`
- **Memory issues?** Reduce `max_features` in `app.py` (line 169)

## How It Works

### Classical ML Model
- Uses TF-IDF vectorization to convert text to numerical features
- Trains a Naive Bayes classifier on the dataset
- Fast inference but limited context understanding

### Transformer Model
- Uses pre-trained RoBERTa sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- Better at understanding context and nuances
- Higher accuracy on complex sentiment analysis

## Project Structure

```
Smart watch sentiment analyser/
├── app.py                 # Flask backend
├── templates/
│   └── index.html        # Frontend HTML
├── static/
│   └── style.css         # Frontend CSS
├── Dataset/
│   └── 7817_1.csv       # Review dataset
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

1. Enter a smartwatch review in the text area
2. Click "Analyze Sentiment" to get predictions from both models
3. Click "Compare Model Performance" to see accuracy comparison on the dataset

## Model Comparison

The transformer model typically shows improved accuracy over the classical ML model due to:
- Better context understanding
- Pre-training on large text corpora
- Attention mechanisms that capture long-range dependencies

