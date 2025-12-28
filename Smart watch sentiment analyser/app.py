from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import shuffle, class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

# Global variables for models
classical_models = {}  # Store all trained models
classical_model = None  # Best model for predictions
classical_vectorizer = None
transformer_pipeline = None
model_results = {}  # Store all model results
best_classical_model = None  # Best model for predictions

def preprocess_text(text):
    """Clean and preprocess text - keep important punctuation for sentiment"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Keep exclamation and question marks as they indicate sentiment
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_transformer_model():
    """Load transformer sentiment model - using high-confidence model"""
    try:
        # Try the best sentiment model first
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        print(f"Loading transformer model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipeline_obj = pipeline(
            "sentiment-analysis", 
            model=model, 
            tokenizer=tokenizer,
            return_all_scores=False,
            device=-1  # Use CPU
        )
        print("Transformer model loaded successfully!")
        return pipeline_obj
    except Exception as e:
        print(f"Error loading primary transformer model: {e}")
        # Try alternative high-confidence models
        try:
            print("Trying alternative model: distilbert-base-uncased-finetuned-sst-2-english")
            return pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=False
            )
        except Exception as e2:
            print(f"Error loading alternative model: {e2}")
            try:
                print("Trying fallback model")
                return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
            except:
                print("Failed to load any transformer model")
                return None

def train_classical_model():
    """Train multiple classical ML models with data balancing, overfitting/underfitting handling"""
    global classical_model, classical_models, classical_vectorizer, model_results
    
    try:
        print("=" * 60)
        print("CLASSICAL ML MODEL TRAINING")
        print("=" * 60)
        
        # Load dataset
        print("\n[1/6] Loading dataset...")
        df = pd.read_csv('Dataset/7817_1.csv')
        print(f"   ✓ Dataset loaded: {len(df)} rows")
        
        # Extract reviews and ratings
        reviews = df['reviews.text'].dropna().astype(str)
        ratings = df['reviews.rating'].dropna()
        
        # Filter to match reviews and ratings
        valid_indices = reviews.index.intersection(ratings.index)
        reviews = reviews.loc[valid_indices]
        ratings = ratings.loc[valid_indices]
        
        print(f"   ✓ Valid reviews with ratings: {len(reviews)}")
        
        # Convert ratings to sentiment labels
        # 1-2: Negative, 3: Neutral, 4-5: Positive
        print("\n[2/6] Converting ratings to sentiment labels...")
        sentiment_labels = []
        for rating in ratings:
            if rating <= 2:
                sentiment_labels.append('negative')
            elif rating == 3:
                sentiment_labels.append('neutral')
            else:
                sentiment_labels.append('positive')
        
        # Preprocess reviews
        print("\n[3/6] Preprocessing reviews...")
        processed_reviews = [preprocess_text(review) for review in reviews]
        
        # Remove empty reviews and ensure minimum length
        valid_data = [(text, label) for text, label in zip(processed_reviews, sentiment_labels) 
                     if text and len(text.strip()) > 2]
        
        if not valid_data:
            print("   ✗ No valid data found for training")
            return False
        
        texts, labels = list(zip(*valid_data))
        print(f"   ✓ Valid processed samples: {len(texts)}")
        
        # Shuffle data
        texts, labels = shuffle(list(texts), list(labels), random_state=42)
        
        # Show class distribution BEFORE balancing
        class_dist = pd.Series(labels).value_counts().to_dict()
        print(f"\n   Class Distribution (BEFORE Balancing):")
        for cls, count in class_dist.items():
            print(f"   - {cls}: {count} ({count/len(labels)*100:.1f}%)")
        
        # Check for class imbalance
        max_class = max(class_dist.values())
        min_class = min(class_dist.values())
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        if imbalance_ratio > 1.5:
            print(f"\n   ⚠ Class Imbalance Detected (Ratio: {imbalance_ratio:.2f})")
            print("   → Applying SMOTE for data balancing...")
        
        # Strict train-test split (BEFORE balancing)
        print("\n[4/7] Creating train-test split (80-20)...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, 
                test_size=0.2, 
                random_state=42, 
                stratify=labels,
                shuffle=True
            )
        except ValueError as ve:
            print(f"   ⚠ Stratification failed: {ve}")
            print("   → Using random split instead...")
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, 
                test_size=0.2, 
                random_state=42,
                shuffle=True
            )
        
        print(f"   ✓ Training samples: {len(X_train)}")
        print(f"   ✓ Test samples: {len(X_test)}")
        
        # Vectorize FIRST (SMOTE needs numeric features)
        print("\n[5/7] Vectorizing text with TF-IDF...")
        
        # Start with more features to prevent underfitting
        initial_features = 20000
        classical_vectorizer = TfidfVectorizer(
            max_features=initial_features,
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,  # Ignore terms in less than 2 documents
            max_df=0.90,  # Ignore terms in more than 90% of documents
            sublinear_tf=True,  # Apply sublinear tf scaling
            norm='l2',  # L2 normalization
            smooth_idf=True  # Smooth IDF weights
        )
        
        print("   → Fitting vectorizer on training data...")
        X_train_vec = classical_vectorizer.fit_transform(X_train)
        X_test_vec = classical_vectorizer.transform(X_test)
        
        print(f"   ✓ Training features: {X_train_vec.shape[1]}")
        print(f"   ✓ Test features: {X_test_vec.shape[1]}")
        
        # Apply SMOTE for balancing (only on training set)
        print("\n[6/7] Applying SMOTE for data balancing...")
        try:
            # Convert sparse matrix to dense for SMOTE
            X_train_dense = X_train_vec.toarray()
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_dense, y_train)
            
            # Convert back to sparse for efficiency
            from scipy.sparse import csr_matrix
            X_train_vec = csr_matrix(X_train_balanced)
            
            balanced_dist = pd.Series(y_train_balanced).value_counts().to_dict()
            print(f"   ✓ Data Balanced! New distribution:")
            for cls, count in balanced_dist.items():
                print(f"      - {cls}: {count} ({count/len(y_train_balanced)*100:.1f}%)")
        except Exception as e:
            print(f"   ⚠ SMOTE failed: {e}")
            print("   → Using class weights instead...")
            X_train_vec = X_train_vec  # Keep original
            y_train_balanced = y_train
        
        # Calculate class weights for models that support it
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train_balanced),
            y=y_train_balanced
        )
        class_weight_dict = dict(zip(np.unique(y_train_balanced), class_weights))
        print(f"   ✓ Class weights: {class_weight_dict}")
        
        # Train Naive Bayes model (only classical model)
        print("\n[7/7] Training Naive Bayes model...")
        
        # Single-model configuration (Naive Bayes only)
        models_config = {
            'Naive Bayes': {
                'model': MultinomialNB(alpha=0.5),  # Higher alpha for regularization
                'use_weights': False
            }
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        results = {}
        classical_models = {}
        
        for name, config in models_config.items():
            print(f"\n   Training {name}...")
            model = config['model']
            
            # Train model
            model.fit(X_train_vec, y_train_balanced)
            
            # Check for overfitting using cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train_vec, y_train_balanced, cv=cv, scoring='f1_macro')
            
            # Test set evaluation
            y_pred = model.predict(X_test_vec)
            y_train_pred = model.predict(X_train_vec[:len(y_train)])  # Predict on original training size
            
            # Calculate train vs test accuracy to detect overfitting
            train_accuracy = accuracy_score(y_train_balanced[:len(y_train)], y_train_pred)
            test_accuracy = accuracy_score(y_test, y_pred)
            overfitting_gap = train_accuracy - test_accuracy
            
            test_f1 = f1_score(y_test, y_pred, average='macro')
            test_precision = precision_score(y_test, y_pred, average='macro')
            test_recall = recall_score(y_test, y_pred, average='macro')
            
            # Store model and results
            classical_models[name] = model
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'overfitting_gap': overfitting_gap,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'status': 'Good' if overfitting_gap < 0.1 else ('Overfitting' if overfitting_gap > 0.15 else 'Slight Overfitting')
            }
            
            print(f"      CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            print(f"      Train Accuracy: {train_accuracy:.4f}")
            print(f"      Test Accuracy: {test_accuracy:.4f}")
            print(f"      Overfitting Gap: {overfitting_gap:.4f} ({results[name]['status']})")
            print(f"      Test F1-Score: {test_f1:.4f}")
            
            # Select best model based on test F1 score
            if test_f1 > best_score:
                best_score = test_f1
                best_model = model
                best_name = name
        
        # Store all results globally
        model_results = results
        
        # Store all models globally
        for name, result in results.items():
            classical_models[name] = result['model']
            print(f"   ✓ Stored model: {name}")
        
        # Use best model
        classical_model = best_model
        print(f"\n   ✓ Best Model Selected: {best_name} (F1: {best_score:.4f})")
        print(f"   ✓ Total models stored: {len(classical_models)}")
        
        # Final evaluation - Show ALL models
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL EVALUATION - ALL MODELS")
        print("=" * 80)
        
        classes = ['negative', 'neutral', 'positive']
        
        # Evaluate each model
        for name, result in results.items():
            print(f"\n{'='*80}")
            print(f"MODEL: {name.upper()}")
            print(f"{'='*80}")
            
            y_pred = result['model'].predict(X_test_vec)
            
            # Detailed metrics
            accuracy = result['test_accuracy']
            f1_macro = result['test_f1']
            precision = result['test_precision']
            recall = result['test_recall']
            
            print(f"\nPerformance Metrics:")
            print(f"  Cross-Validation F1:  {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
            print(f"  Train Accuracy:      {result['train_accuracy']:.4f}")
            print(f"  Test Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Test F1-Score:       {f1_macro:.4f}")
            print(f"  Test Precision:      {precision:.4f}")
            print(f"  Test Recall:         {recall:.4f}")
            print(f"  Overfitting Status:  {result['status']} (Gap: {result['overfitting_gap']:.4f})")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"              Neg  Neu  Pos")
            for i, cls in enumerate(classes):
                print(f"Actual {cls[:3]:>5}  {cm[i][0]:4d} {cm[i][1]:4d} {cm[i][2]:4d}")
        
        # Best model detailed report
        print(f"\n{'='*80}")
        print(f"BEST MODEL DETAILED REPORT: {best_name.upper()}")
        print(f"{'='*80}")
        
        y_pred = classical_model.predict(X_test_vec)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=classes, digits=4))
        
        print(f"\nPer-Class Metrics:")
        for i, cls in enumerate(classes):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(3) if j != i)
            fn = sum(cm[i][j] for j in range(3) if j != i)
            
            precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_cls = 2 * (precision_cls * recall_cls) / (precision_cls + recall_cls) if (precision_cls + recall_cls) > 0 else 0
            
            print(f"  {cls:>8}: Precision={precision_cls:.4f}, Recall={recall_cls:.4f}, F1={f1_cls:.4f}")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY - ALL MODELS READY")
        print("=" * 80 + "\n")
        
        return True
    except Exception as e:
        print(f"\n✗ Error training classical model: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_classical(text):
    """Predict sentiment using classical ML model"""
    if classical_model is None or classical_vectorizer is None:
        print("Warning: Classical model not trained yet")
        return None, 0.0
    
    processed = preprocess_text(text)
    if not processed:
        return 'neutral', 0.5
    
    try:
        text_vec = classical_vectorizer.transform([processed])
        
        # Check if vectorizer has any features
        if text_vec.shape[1] == 0:
            print("Warning: No features extracted from text")
            return 'neutral', 0.5
        
        prediction = classical_model.predict(text_vec)[0]
        probabilities = classical_model.predict_proba(text_vec)[0]
        
        # Get confidence
        classes = classical_model.classes_
        conf_idx = list(classes).index(prediction)
        confidence = probabilities[conf_idx]
        
        # Ensure minimum confidence threshold
        if confidence < 0.3:
            # For very low confidence, check common sentiment words
            text_lower = processed.lower()
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointed', 'poor', 'waste', 'disappointing']
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'perfect', 'fantastic', 'awesome']
            
            if any(word in text_lower for word in negative_words):
                return 'negative', max(confidence, 0.6)
            elif any(word in text_lower for word in positive_words):
                return 'positive', max(confidence, 0.6)
        
        return prediction, confidence
    except Exception as e:
        print(f"Error in classical prediction: {e}")
        return 'neutral', 0.5

def split_into_sentences(text):
    """Split text into sentences"""
    try:
        # Download punkt tokenizer if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        sentences = sent_tokenize(text)
        # Filter out very short sentences (likely artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences if sentences else [text]
    except Exception as e:
        print(f"Error splitting sentences: {e}")
        # Fallback: split by periods
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        return sentences if sentences else [text]

def predict_transformer(text):
    """Predict sentiment using transformer model - returns overall and sentence-level"""
    if transformer_pipeline is None:
        return None, 0.0, []
    
    try:
        # Get overall sentiment
        max_length = 512
        text_for_overall = text[:max_length] if len(text) > max_length else text
        
        # For very short text, add context to improve confidence
        if len(text_for_overall.strip()) < 10:
            # Add context for single words
            text_for_overall = f"This product is {text_for_overall}."
        
        overall_result = transformer_pipeline(text_for_overall)[0]
        
        # Map transformer output to our labels
        label = overall_result['label'].lower()
        score = overall_result['score']
        
        # Boost confidence for clear predictions
        if score < 0.6:
            # If confidence is low, check if it's a very clear case
            text_lower = text.lower().strip()
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointed', 'poor', 'waste']
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'perfect', 'fantastic']
            
            if any(word in text_lower for word in negative_words):
                if 'negative' in label or 'neg' in label or 'négatif' in label:
                    score = max(score, 0.75)  # Boost confidence
            elif any(word in text_lower for word in positive_words):
                if 'positive' in label or 'pos' in label or 'positif' in label:
                    score = max(score, 0.75)  # Boost confidence
        
        # Convert transformer labels to our format
        if 'positive' in label or 'pos' in label or 'positif' in label or 'positives' in label:
            overall_sentiment = 'positive'
        elif 'negative' in label or 'neg' in label or 'négatif' in label or 'negatives' in label:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Get sentence-level analysis
        sentences = split_into_sentences(text)
        sentence_analyses = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 5:  # Skip very short sentences
                continue
                
            try:
                # Truncate sentence if too long
                sentence_text = sentence[:max_length] if len(sentence) > max_length else sentence
                
                # Add context for very short sentences
                if len(sentence_text.strip()) < 10:
                    sentence_text = f"This product is {sentence_text}."
                
                sent_result = transformer_pipeline(sentence_text)[0]
                
                sent_label = sent_result['label'].lower()
                sent_score = sent_result['score']
                
                # Boost confidence for sentence-level too
                if sent_score < 0.6:
                    sent_lower = sentence.lower().strip()
                    if any(word in sent_lower for word in ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate']):
                        if 'negative' in sent_label or 'neg' in sent_label:
                            sent_score = max(sent_score, 0.75)
                    elif any(word in sent_lower for word in ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love']):
                        if 'positive' in sent_label or 'pos' in sent_label:
                            sent_score = max(sent_score, 0.75)
                
                # Convert to our format
                if 'positive' in sent_label or 'pos' in sent_label or 'positif' in sent_label:
                    sent_sentiment = 'positive'
                elif 'negative' in sent_label or 'neg' in sent_label or 'négatif' in sent_label:
                    sent_sentiment = 'negative'
                else:
                    sent_sentiment = 'neutral'
                
                sentence_analyses.append({
                    'sentence': sentence.strip(),
                    'sentiment': sent_sentiment,
                    'confidence': round(sent_score * 100, 2)
                })
            except Exception as e:
                print(f"Error analyzing sentence: {e}")
                continue
        
        return overall_sentiment, score, sentence_analyses
    except Exception as e:
        print(f"Error in transformer prediction: {e}")
        import traceback
        traceback.print_exc()
        return 'neutral', 0.5, []

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    """Check model status"""
    return jsonify({
        'classical_models_count': len(classical_models),
        'classical_models': list(classical_models.keys()),
        'best_model_loaded': classical_model is not None,
        'vectorizer_loaded': classical_vectorizer is not None,
        'transformer_loaded': transformer_pipeline is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment of input text"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"\n[DEBUG] Analyzing text: {text[:50]}...")
        print(f"[DEBUG] Classical models available: {list(classical_models.keys())}")
        print(f"[DEBUG] Classical model (best): {classical_model is not None}")
        print(f"[DEBUG] Vectorizer: {classical_vectorizer is not None}")
        print(f"[DEBUG] Transformer: {transformer_pipeline is not None}")
        
        # Get predictions from all classical ML models
        all_ml_predictions = {}
        if classical_models and len(classical_models) > 0:
            for name, model in classical_models.items():
                try:
                    sentiment, confidence = predict_with_model(text, model)
                    all_ml_predictions[name] = {
                        'sentiment': sentiment if sentiment else 'neutral',
                        'confidence': round(confidence * 100, 2) if confidence else 0
                    }
                    print(f"[DEBUG] {name}: {sentiment} ({confidence:.2f})")
                except Exception as e:
                    print(f"[DEBUG] Error with {name}: {e}")
                    all_ml_predictions[name] = {
                        'sentiment': 'neutral',
                        'confidence': 0
                    }
        else:
            print("[DEBUG] No classical models available, using best model only")
        
        # Get best model prediction
        classical_sentiment, classical_confidence = predict_classical(text)
        if not classical_sentiment:
            classical_sentiment = 'neutral'
        if not classical_confidence:
            classical_confidence = 0.5
        
        print(f"[DEBUG] Best model: {classical_sentiment} ({classical_confidence:.2f})")
        
        # Get transformer prediction
        transformer_sentiment, transformer_confidence, sentence_analyses = predict_transformer(text)
        if not transformer_sentiment:
            transformer_sentiment = 'neutral'
        if not transformer_confidence:
            transformer_confidence = 0.5
        
        print(f"[DEBUG] Transformer: {transformer_sentiment} ({transformer_confidence:.2f})")
        
        result = {
            'text': text,
            'classical_ml': {
                'sentiment': classical_sentiment,
                'confidence': round(classical_confidence * 100, 2)
            },
            'all_ml_models': all_ml_predictions if all_ml_predictions else {
                'Best Model': {
                    'sentiment': classical_sentiment,
                    'confidence': round(classical_confidence * 100, 2)
                }
            },
            'transformer': {
                'sentiment': transformer_sentiment,
                'confidence': round(transformer_confidence * 100, 2)
            },
            'sentences': sentence_analyses if sentence_analyses else []
        }
        
        print(f"[DEBUG] Returning result with {len(all_ml_predictions)} ML models")
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Analyze endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def predict_with_model(text, model):
    """Predict sentiment using a specific model"""
    if model is None or classical_vectorizer is None:
        return None, 0.0
    
    processed = preprocess_text(text)
    if not processed:
        return 'neutral', 0.5
    
    try:
        text_vec = classical_vectorizer.transform([processed])
        if text_vec.shape[1] == 0:
            return 'neutral', 0.5
        
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]
        
        classes = model.classes_
        conf_idx = list(classes).index(prediction)
        confidence = probabilities[conf_idx]
        
        return prediction, confidence
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return 'neutral', 0.5

if __name__ == '__main__':
    print("Initializing models...")
    print("Training classical ML model...")
    train_classical_model()
    print("Loading transformer model...")
    transformer_pipeline = load_transformer_model()
    if transformer_pipeline:
        print("Transformer model loaded successfully!")
    else:
        print("Warning: Could not load transformer model")
    
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

