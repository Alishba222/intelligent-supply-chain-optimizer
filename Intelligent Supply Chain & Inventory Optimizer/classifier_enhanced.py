# Enhanced Machine Learning Classifier for Document Classification
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Fallback to keyword-based classifier if model not trained
keyword_classifier = {
    "weather": ["weather", "rain", "monsoon", "storm", "fog"],
    "sports": ["cricket", "match", "game", "tournament", "stadium"],
    "emotional": ["angry", "sad", "distress", "fatigue", "complaint"],
    "incident": ["accident", "blocked", "incident", "crash", "stoppage"],
    "adventure": ["adventure", "expedition", "off-road", "terrain", "mountain"],
    "religious": ["religious", "holiday", "festival", "eid", "diwali", "prayer"],
    "supplier": ["supplier", "vendor", "shortage", "halt", "production"],
    "regulatory": ["regulatory", "customs", "compliance", "inspection", "permit"],
    "maintenance": ["maintenance", "service", "repair", "breakdown", "equipment"],
}

MODEL_PATH = "classifier_model.pkl"

def train_classifier(texts, labels):
    """Train ML classifier on labeled text data"""
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100, stop_words='english')),
        ('clf', MultinomialNB())
    ])
    
    pipeline.fit(texts, labels)
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    
    return pipeline

def load_classifier():
    """Load pre-trained classifier if exists"""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def classify_document_ml(text, model=None):
    """ML-based document classification with confidence score"""
    if model is None:
        model = load_classifier()
    
    if model is not None:
        prediction = model.predict([text])[0]
        confidence = model.predict_proba([text])[0].max()
        return prediction, confidence
    
    return None, 0.0

def classify_document(text, use_ml=True):
    """
    Main classification function - tries ML first, falls back to keywords
    
    Returns:
        category (str): Document category (CAT_*)
        confidence (float): Confidence score 0.0-1.0
    """
    text_lower = text.lower()
    
    # Try ML classifier first if available
    if use_ml:
        ml_category, ml_confidence = classify_document_ml(text_lower)
        if ml_category and ml_confidence > 0.5:
            category_map = {
                "weather": "CAT_WEATHER",
                "sports": "CAT_SPORTS",
                "emotional": "CAT_EMOTIONAL",
                "incident": "CAT_INCIDENT",
                "adventure": "CAT_ADVENTURE",
                "religious": "CAT_RELIGIOUS",
                "supplier": "CAT_SUPPLIER",
                "regulatory": "CAT_REGULATORY",
                "maintenance": "CAT_MAINTENANCE",
            }
            return category_map.get(ml_category, "CAT_GENERAL"), ml_confidence
    
    # Fallback to keyword-based classification with confidence scoring
    max_matches = 0
    best_category = "CAT_GENERAL"
    
    for category, keywords in keyword_classifier.items():
        matches = sum(1 for word in keywords if word in text_lower)
        if matches > max_matches:
            max_matches = matches
            best_category = f"CAT_{category.upper()}"
    
    # Calculate confidence based on keyword matches
    total_keywords = sum(len(words) for words in keyword_classifier.values())
    confidence = min(max_matches / 3.0, 1.0) if max_matches > 0 else 0.1
    
    return best_category, confidence
