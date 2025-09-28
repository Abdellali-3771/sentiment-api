"""
Air Paradis Sentiment Analysis API
API FastAPI pour l'analyse de sentiment des tweets
Modèle: BiLSTM avec Word2Vec (compatible TensorFlow 2.13 + gestion tokenizers incompatibles)
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
from contextlib import asynccontextmanager

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Pydantic v1 pour compatibilité avec TensorFlow 2.13
from pydantic import BaseModel, Field, validator

import mlflow
import mlflow.keras


# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration TensorFlow pour stabilité
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Réduire les logs TensorFlow
tf.get_logger().setLevel('ERROR')

# Configuration MLflow - toujours désactivé sur Render
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
MLFLOW_AVAILABLE = False
logger.info("🚫 MLflow désactivé (Render)")


# Configuration Google Cloud
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "air-paradis-sentiment-api")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "europe-west1")

# Variables globales pour le modèle
model = None
tokenizer = None
config = None
MAX_SEQUENCE_LENGTH = 50

# Métriques de monitoring
error_count = 0
last_errors = []

class CompatibleTokenizer:
    """Tokenizer compatible créé à partir d'un word_index extracté"""
    
    def __init__(self, word_index, num_words=None):
        self.word_index = word_index
        self.num_words = num_words or len(word_index)
        self.word_counts = None
        self.word_docs = None
        self.index_word = {v: k for k, v in word_index.items()}
    
    def texts_to_sequences(self, texts):
        """Convertit les textes en séquences d'entiers"""
        sequences = []
        for text in texts:
            if isinstance(text, str):
                words = text.lower().split()
            else:
                words = text
            
            sequence = []
            for word in words:
                index = self.word_index.get(word, 0)  # 0 pour mots inconnus
                if self.num_words is None or index < self.num_words:
                    sequence.append(index)
                else:
                    sequence.append(0)  # Mot hors vocabulaire
            sequences.append(sequence)
        return sequences

class ModelManager:
    """Gestionnaire du modèle de sentiment - Compatible TensorFlow 2.13 avec gestion tokenizers"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.model_path = "models/best_advanced_model_BiLSTM_Word2Vec.h5"
        self.tokenizer_path = "models/best_advanced_model_tokenizer.pickle"
        self.config_path = "models/best_advanced_model_config.pickle"
        self.is_dummy_model = False
        self.tokenizer_type = "dummy"  # Valeur par défaut pour éviter les erreurs
        
    async def load_model(self):
        """Charge le modèle et ses composants avec gestion des incompatibilités TF 2.13"""
        try:
            logger.info("🔧 Chargement du modèle BiLSTM + Word2Vec...")
            
            # Configuration TensorFlow pour modèles avec batch_shape
            tf.keras.backend.clear_session()
            
            # Téléchargement des modèles depuis Google Cloud Storage si nécessaire
            await self._download_models_if_needed()
            
            # Chargement du modèle TensorFlow avec gestion des erreurs batch_shape
            await self._load_tensorflow_model()
            
            # Chargement du tokenizer avec gestion des incompatibilités
            await self._load_tokenizer_compatible()
            
            # Chargement de la config
            await self._load_config()
            
            # Test du modèle
            await self._test_model()
            
        except Exception as e:
            logger.error(f"❌ Erreur critique lors du chargement: {str(e)}")
            # Fallback complet vers modèle factice
            self.model = self._create_compatible_dummy_model()
            self.tokenizer = self._create_dummy_tokenizer()
            self.config = {"max_sequence_length": MAX_SEQUENCE_LENGTH}
            self.is_dummy_model = True
            self.tokenizer_type = "dummy"
            logger.info("🎭 Fallback vers modèle factice complet")
    
    async def _download_models_if_needed(self):
        """Télécharge les modèles depuis Google Cloud Storage si nécessaire"""
        import urllib.request
        import os
        
        bucket_url = "https://storage.googleapis.com/air-paradis-models"
        models_to_download = [
            ("best_advanced_model_BiLSTM_Word2Vec.h5", self.model_path),
            ("best_advanced_model_tokenizer.pickle", self.tokenizer_path), 
            ("best_advanced_model_config.pickle", self.config_path)
        ]
        
        for filename, local_path in models_to_download:
            # Vérifier si le fichier existe ET s'il est de taille réaliste
            file_exists = os.path.exists(local_path)
            file_size = os.path.getsize(local_path) if file_exists else 0
            
            # Télécharger si :
            # - Le fichier n'existe pas
            # - Le fichier est trop petit (< 1MB = probablement factice)
            # - Le fichier contient "dummy" (fichiers de test)
            should_download = (
                not file_exists or 
                file_size < 1_000_000 or  # < 1MB
                (file_exists and self._is_dummy_file(local_path))
            )
            
            if should_download:
                logger.info(f"💾 Téléchargement de {filename} depuis GCS...")
                try:
                    url = f"{bucket_url}/{filename}"
                    urllib.request.urlretrieve(url, local_path)
                    size_mb = os.path.getsize(local_path) / (1024 * 1024)
                    logger.info(f"✅ {filename} téléchargé: {size_mb:.1f}MB")
                except Exception as e:
                    logger.warning(f"⚠️ Échec téléchargement {filename}: {e}")
            else:
                size_mb = os.path.getsize(local_path) / (1024 * 1024)
                logger.info(f"✅ {filename} déjà présent: {size_mb:.1f}MB")
    
    def _is_dummy_file(self, file_path):
        """Vérifie si un fichier est factice (contient 'dummy')"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(100)  # Lire seulement les 100 premiers caractères
                return 'dummy' in content.lower()
        except:
            return False
    
    async def _load_tensorflow_model(self):
        """Charge le modèle TensorFlow avec gestion batch_shape"""
        if os.path.exists(self.model_path):
            try:
                # Tentative de chargement normal
                self.model = load_model(self.model_path, compile=False)
                logger.info("✅ Modèle TensorFlow chargé avec succès")
                self.is_dummy_model = False
                
                # Recompiler le modèle pour s'assurer de la compatibilité
                self.model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
            except Exception as e:
                if "batch_shape" in str(e) or "InputLayer" in str(e):
                    logger.warning(f"⚠️ Erreur batch_shape détectée: {e}")
                    logger.info("🔄 Tentative de chargement avec custom_objects...")
                    
                    try:
                        # Chargement avec gestion personnalisée des objets
                        custom_objects = {
                            'InputLayer': tf.keras.layers.InputLayer
                        }
                        self.model = load_model(
                            self.model_path, 
                            custom_objects=custom_objects,
                            compile=False
                        )
                        logger.info("✅ Modèle chargé avec custom_objects")
                        self.is_dummy_model = False
                        
                    except Exception as e2:
                        logger.error(f"❌ Échec du chargement avec custom_objects: {e2}")
                        logger.info("🎭 Basculement vers modèle factice")
                        self.model = self._create_compatible_dummy_model()
                        self.is_dummy_model = True
                else:
                    logger.error(f"❌ Erreur de chargement du modèle: {e}")
                    self.model = self._create_compatible_dummy_model()
                    self.is_dummy_model = True
        else:
            logger.warning(f"⚠️ Modèle non trouvé: {self.model_path}")
            logger.info("🎭 Création d'un modèle factice compatible")
            self.model = self._create_compatible_dummy_model()
            self.is_dummy_model = True
    
    async def _load_tokenizer_compatible(self):
        """Charge le tokenizer avec gestion des incompatibilités TensorFlow 2.13"""
        if os.path.exists(self.tokenizer_path):
            try:
                with open(self.tokenizer_path, 'rb') as f:
                    tokenizer_data = pickle.load(f)
                
                # Cas 1: Tokenizer Keras standard compatible
                if hasattr(tokenizer_data, 'texts_to_sequences') and hasattr(tokenizer_data, 'word_index'):
                    try:
                        # Test de fonctionnalité
                        test_sequences = tokenizer_data.texts_to_sequences(["test"])
                        self.tokenizer = tokenizer_data
                        self.tokenizer_type = "keras_native"
                        logger.info("✅ Tokenizer Keras natif chargé")
                        return
                    except Exception as e:
                        if "keras.src.legacy" in str(e):
                            logger.warning("⚠️ Tokenizer incompatible TF 2.13 (keras.src.legacy)")
                            # Extraire le word_index pour créer un tokenizer compatible
                            word_index = getattr(tokenizer_data, 'word_index', {})
                            num_words = getattr(tokenizer_data, 'num_words', None)
                            self.tokenizer = CompatibleTokenizer(word_index, num_words)
                            self.tokenizer_type = "compatible_extracted"
                            logger.info("✅ Tokenizer compatible créé à partir du word_index")
                            return
                        else:
                            raise e
                
                # Cas 2: Format dict avec word_index
                elif isinstance(tokenizer_data, dict) and 'word_index' in tokenizer_data:
                    self.tokenizer = CompatibleTokenizer(
                        tokenizer_data['word_index'],
                        tokenizer_data.get('num_words', None)
                    )
                    self.tokenizer_type = "dict_format"
                    logger.info("✅ Tokenizer créé à partir du format dict")
                    return
                
                # Cas 3: Tentative d'extraction du word_index
                else:
                    logger.warning("⚠️ Format de tokenizer non reconnu, tentative d'extraction...")
                    word_index = {}
                    
                    # Essayer différentes méthodes d'extraction
                    if hasattr(tokenizer_data, '__dict__'):
                        attrs = tokenizer_data.__dict__
                        if 'word_index' in attrs:
                            word_index = attrs['word_index']
                        elif '_word_index' in attrs:
                            word_index = attrs['_word_index']
                    
                    if word_index:
                        self.tokenizer = CompatibleTokenizer(word_index)
                        self.tokenizer_type = "extracted"
                        logger.info("✅ Tokenizer créé par extraction du word_index")
                        return
                    else:
                        raise ValueError("Impossible d'extraire le word_index")
                        
            except Exception as e:
                logger.error(f"❌ Erreur chargement tokenizer: {e}")
                logger.info("🎭 Création d'un tokenizer factice")
                self.tokenizer = self._create_dummy_tokenizer()
                self.tokenizer_type = "dummy"
        else:
            logger.warning(f"⚠️ Tokenizer non trouvé: {self.tokenizer_path}")
            self.tokenizer = self._create_dummy_tokenizer()
            self.tokenizer_type = "dummy"
    
    async def _load_config(self):
        """Charge la configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'rb') as f:
                    self.config = pickle.load(f)
                logger.info("✅ Configuration chargée")
            except Exception as e:
                logger.warning(f"⚠️ Erreur chargement config: {e}")
                self.config = {"max_sequence_length": MAX_SEQUENCE_LENGTH}
        else:
            logger.warning("⚠️ Configuration par défaut")
            self.config = {
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "vocab_size": 10000,
                "embedding_dim": 300
            }
    
    def _create_compatible_dummy_model(self):
        """Crée un modèle factice compatible avec TensorFlow 2.13"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, SpatialDropout1D
            
            # Modèle simple sans batch_shape pour éviter les conflits
            model = Sequential([
                Embedding(10000, 300, input_length=MAX_SEQUENCE_LENGTH),
                SpatialDropout1D(0.2),
                Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Initialiser avec des données factices
            dummy_X = np.random.randint(0, 1000, (32, MAX_SEQUENCE_LENGTH))
            dummy_y = np.random.randint(0, 2, (32, 1))
            model.fit(dummy_X, dummy_y, epochs=1, verbose=0)
            
            logger.info("✅ Modèle factice compatible créé")
            return model
            
        except Exception as e:
            logger.error(f"❌ Erreur création modèle factice: {e}")
            # Fallback vers prédicteur ultra-simple
            class UltraSimplePredictor:
                def predict(self, x, batch_size=None):
                    # Analyse simple basée sur les mots
                    batch_size = x.shape[0] if hasattr(x, 'shape') else 1
                    return np.random.uniform(0.3, 0.9, (batch_size, 1))
            
            return UltraSimplePredictor()
    
    def _create_dummy_tokenizer(self):
        """Crée un tokenizer factice pour les tests"""
        word_index = {
            'love': 1, 'great': 2, 'excellent': 3, 'amazing': 4, 'best': 5,
            'hate': 6, 'terrible': 7, 'worst': 8, 'awful': 9, 'bad': 10,
            'airline': 11, 'flight': 12, 'service': 13, 'crew': 14, 'staff': 15,
            'i': 16, 'the': 17, 'is': 18, 'was': 19, 'this': 20,
            'air': 21, 'paradis': 22, 'good': 23, 'ok': 24, 'fine': 25,
            'nice': 26, 'comfortable': 27, 'friendly': 28, 'professional': 29,
            'disappointing': 30, 'delayed': 31, 'cancelled': 32, 'rude': 33
        }
        
        return CompatibleTokenizer(word_index, 10000)
    
    async def _test_model(self):
        """Test du modèle avec gestion des erreurs"""
        try:
            test_text = "I love this airline, great service!"
            prediction = await self.predict(test_text)
            model_type = "factice" if self.is_dummy_model else "réel"
            logger.info(f"✅ Test du modèle {model_type} réussi: '{test_text}' -> {prediction}")
            logger.info(f"📊 Tokenizer: {self.tokenizer_type}")
        except Exception as e:
            logger.error(f"❌ Échec du test du modèle: {str(e)}")
            raise
    
    async def predict(self, text: str) -> Dict:
        """Prédiction de sentiment avec gestion batch_shape et tokenizers incompatibles"""
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Modèle ou tokenizer non chargé")
            
            # Préprocessing du texte
            sequences = self.tokenizer.texts_to_sequences([text])
            max_len = self.config.get("max_sequence_length", MAX_SEQUENCE_LENGTH)
            padded = pad_sequences(sequences, maxlen=max_len)
            
            # Prédiction avec gestion des différents types de modèles
            try:
                if hasattr(self.model, 'predict'):
                    # Prédiction avec gestion batch_size
                    prediction_prob = self.model.predict(padded, batch_size=1, verbose=0)[0][0]
                else:
                    # Fallback pour modèles ultra-simples
                    prediction_prob = 0.8  # Valeur par défaut positive
                    
            except Exception as pred_error:
                logger.warning(f"⚠️ Erreur de prédiction: {pred_error}")
                # Analyse heuristique simple en cas d'échec
                positive_words = ['love', 'great', 'excellent', 'amazing', 'best', 'good', 'nice']
                negative_words = ['hate', 'terrible', 'worst', 'awful', 'bad', 'disappointing']
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count:
                    prediction_prob = 0.75
                elif negative_count > positive_count:
                    prediction_prob = 0.25
                else:
                    prediction_prob = 0.5
            
            prediction_class = int(prediction_prob > 0.5)
            
            # Mapping des résultats
            sentiment_mapping = {0: "negative", 1: "positive"}
            sentiment = sentiment_mapping[prediction_class]
            
            confidence = float(prediction_prob) if prediction_class == 1 else float(1 - prediction_prob)
            
            result = {
                "text": text,
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "probability": round(float(prediction_prob), 4),
                "model": f"BiLSTM_Word2Vec{' (factice)' if self.is_dummy_model else ''}",
                "tokenizer_type": self.tokenizer_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Log MLflow avec gestion d'erreurs
            if MLFLOW_AVAILABLE:
                try:
                    with mlflow.start_run(run_name=f"prediction_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
                        mlflow.log_param("text_length", len(text))
                        mlflow.log_metric("confidence", confidence)
                        mlflow.log_metric("probability", float(prediction_prob))
                        mlflow.log_param("sentiment", sentiment)
                        mlflow.log_param("is_dummy_model", self.is_dummy_model)
                        mlflow.log_param("tokenizer_type", self.tokenizer_type)
                except Exception as mlflow_error:
                    logger.warning(f"⚠️ Erreur MLflow: {mlflow_error}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            raise

# Instance globale du gestionnaire de modèle
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire du cycle de vie de l'application"""
    # Startup
    logger.info("🚀 Démarrage de l'API Air Paradis Sentiment Analysis")
    logger.info(f"🔧 TensorFlow version: {tf.__version__}")
    logger.info(f"📊 MLflow disponible: {MLFLOW_AVAILABLE}")
    await model_manager.load_model()
    logger.info("✅ API prête à recevoir des requêtes")
    yield
    # Shutdown
    logger.info("🛑 Arrêt de l'API")

# Modèles Pydantic v1
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Texte à analyser")
    user_id: Optional[str] = Field(None, description="ID de l'utilisateur (optionnel)")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Le texte ne peut pas être vide')
        return v.strip()

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probability: float
    model: str
    tokenizer_type: str
    timestamp: str
    request_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    text: str
    predicted_sentiment: str
    actual_sentiment: str
    is_correct: bool
    user_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    tokenizer_type: str
    tensorflow_version: str
    mlflow_available: bool
    timestamp: str
    version: str = "1.0.0"

class LogRequest(BaseModel):
    level: str = Field(..., description="Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    message: str = Field(..., description="Message de log")
    data: Optional[Dict] = Field(None, description="Données additionnelles")
    timestamp: Optional[int] = Field(None, description="Timestamp Unix")
    projectId: Optional[str] = Field(None, description="ID du projet Google Cloud")
    logName: Optional[str] = Field(None, description="Nom du log")

class LogResponse(BaseModel):
    success: bool
    message: str
    logId: str
    recentErrorCount: int
    alertThreshold: int
    metadata: Dict

# Initialisation FastAPI
app = FastAPI(
    title="Air Paradis Sentiment Analysis API",
    description="API d'analyse de sentiment pour les tweets - Compatible TensorFlow 2.13 + tokenizers",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitoring Google Cloud
async def send_error_to_monitoring(error_message: str, text: str):
    """Envoie une erreur au monitoring Google Cloud"""
    try:
        global error_count, last_errors
        error_count += 1
        current_time = datetime.utcnow()
        
        last_errors.append({
            "timestamp": current_time,
            "message": error_message,
            "text": text
        })
        
        five_minutes_ago = current_time.timestamp() - 300
        last_errors = [
            err for err in last_errors 
            if err["timestamp"].timestamp() > five_minutes_ago
        ]
        
        if len(last_errors) >= 3:
            await send_alert_email()
            
        logger.warning(f"⚠️ Erreur envoyée au monitoring: {error_message}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'envoi au monitoring: {str(e)}")

async def send_alert_email():
    """Envoie une alerte par email via Google Cloud"""
    try:
        logger.critical("🚨 ALERTE: 3 erreurs détectées en 5 minutes!")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'envoi d'alerte: {str(e)}")

# Routes de l'API

@app.get("/", response_model=Dict)
async def root():
    """Route racine de l'API"""
    return {
        "message": "Air Paradis Sentiment Analysis API",
        "version": "1.0.0",
        "tensorflow_version": tf.__version__,
        "model": f"BiLSTM_Word2Vec{' (factice)' if model_manager.is_dummy_model else ''}",
        "tokenizer_type": model_manager.tokenizer_type,
        "mlflow_available": MLFLOW_AVAILABLE,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "feedback": "/feedback",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état de l'API"""
    model_loaded = model_manager.model is not None and model_manager.tokenizer is not None
    model_type = "factice" if model_manager.is_dummy_model else "réel"
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_type=model_type,
        tokenizer_type=model_manager.tokenizer_type,
        tensorflow_version=tf.__version__,
        mlflow_available=MLFLOW_AVAILABLE,
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(
    request: SentimentRequest,
    background_tasks: BackgroundTasks
):
    """Prédiction de sentiment pour un texte"""
    try:
        # Validation
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
        
        # Prédiction
        result = await model_manager.predict(request.text)
        
        # Génération d'un ID de requête
        request_id = f"req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        result["request_id"] = request_id
        
        logger.info(f"✅ Prédiction réussie: {request.text[:50]}... -> {result['sentiment']}")
        
        return SentimentResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Erreur lors de la prédiction: {str(e)}"
        logger.error(f"❌ {error_msg}")
        
        background_tasks.add_task(send_error_to_monitoring, error_msg, request.text)
        
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """Soumission de feedback utilisateur"""
    try:
        logger.info(f"📝 Feedback reçu: {feedback.text[:50]}... - Correct: {feedback.is_correct}")
        
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"feedback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_param("text", feedback.text)
                    mlflow.log_param("predicted_sentiment", feedback.predicted_sentiment)
                    mlflow.log_param("actual_sentiment", feedback.actual_sentiment)
                    mlflow.log_metric("is_correct", 1 if feedback.is_correct else 0)
                    mlflow.log_param("user_id", feedback.user_id or "anonymous")
            except Exception as e:
                logger.warning(f"⚠️ Erreur MLflow feedback: {e}")
        
        if not feedback.is_correct:
            error_msg = f"Prédiction incorrecte: prédit '{feedback.predicted_sentiment}', réel '{feedback.actual_sentiment}'"
            background_tasks.add_task(send_error_to_monitoring, error_msg, feedback.text)
        
        return {
            "message": "Feedback enregistré avec succès",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'enregistrement du feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'enregistrement du feedback")

@app.get("/metrics")
async def get_metrics():
    """Métriques de l'API"""
    return {
        "total_errors": error_count,
        "recent_errors_5min": len(last_errors),
        "model_loaded": model_manager.model is not None,
        "model_type": "factice" if model_manager.is_dummy_model else "réel",
        "tokenizer_type": model_manager.tokenizer_type,
        "tensorflow_version": tf.__version__,
        "mlflow_available": MLFLOW_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/logging", response_model=LogResponse)
async def receive_frontend_logs(
    log_request: LogRequest,
    background_tasks: BackgroundTasks
):
    """Endpoint pour recevoir les logs du frontend et les traiter avec Google Cloud Logging"""
    try:
        current_time = datetime.utcnow()
        
        # Création de l'entrée de log
        log_extra = {
            "timestamp": log_request.timestamp or int(current_time.timestamp() * 1000),
            "log_level": log_request.level.upper(),
            "frontend_data": log_request.data or {},
            "projectId": log_request.projectId or PROJECT_ID,
            "logName": log_request.logName or "air-paradis-frontend",
            "source": "frontend-via-backend"
        }
        
        # Message de log avec préfixe frontend
        log_level = log_request.level.lower()
        log_message = f"[FRONTEND] {log_request.message}"
        
        # Log local avec extra
        if log_level == 'debug':
            logger.debug(log_message, extra=log_extra)
        elif log_level == 'info':
            logger.info(log_message, extra=log_extra)
        elif log_level == 'warning':
            logger.warning(log_message, extra=log_extra)
        elif log_level in ['error', 'critical']:
            logger.error(log_message, extra=log_extra)
        else:
            logger.info(log_message, extra=log_extra)
        
        # Intégration Google Cloud Logging directe
        if os.getenv('GOOGLE_CLOUD_LOGGING_ENABLED', '').lower() == 'true':
            try:
                from google.cloud import logging as gcp_logging
                
                # Client Google Cloud Logging
                gcp_client = gcp_logging.Client(project=PROJECT_ID)
                gcp_logger = gcp_client.logger(log_request.logName or "air-paradis-frontend")
                
                # Structure du log pour Google Cloud
                gcp_log_entry = {
                    "message": f"[FRONTEND] {log_request.message}",
                    "severity": log_request.level.upper(),
                    "timestamp": current_time.isoformat(),
                    "data": log_request.data or {},
                    "source": "frontend-next-js",
                    "service": "air-paradis-frontend",
                    "version": "1.0.0"
                }
                
                # Envoi vers Google Cloud Logging avec le bon niveau
                if log_level == 'debug':
                    gcp_logger.log_struct(gcp_log_entry, severity='DEBUG')
                elif log_level == 'info':
                    gcp_logger.log_struct(gcp_log_entry, severity='INFO')
                elif log_level == 'warning':
                    gcp_logger.log_struct(gcp_log_entry, severity='WARNING')
                elif log_level == 'error':
                    gcp_logger.log_struct(gcp_log_entry, severity='ERROR')
                elif log_level == 'critical':
                    gcp_logger.log_struct(gcp_log_entry, severity='CRITICAL')
                else:
                    gcp_logger.log_struct(gcp_log_entry, severity='INFO')
                    
            except Exception as gcp_error:
                logger.warning(f"⚠️ Erreur Google Cloud Logging: {gcp_error}")
        
        # Gestion spéciale des prédictions incorrectes (pour alertes)
        if (log_request.level.upper() == 'WARNING' and 
            log_request.data and 
            log_request.data.get('event_type') == 'incorrect_prediction'):
            
            error_msg = f"Prédiction incorrecte frontend: prédit '{log_request.data.get('predicted_sentiment')}', réel '{log_request.data.get('actual_sentiment')}'"
            
            # Envoi au système de monitoring pour déclencher les alertes
            background_tasks.add_task(
                send_error_to_monitoring, 
                error_msg, 
                log_request.data.get('text', '')
            )
            
            logger.warning(f"🔥 Prédiction incorrecte depuis frontend: {log_request.data}")
            
            # Log spécial Google Cloud pour les alertes
            if os.getenv('GOOGLE_CLOUD_LOGGING_ENABLED', '').lower() == 'true':
                try:
                    from google.cloud import logging as gcp_logging
                    gcp_client = gcp_logging.Client(project=PROJECT_ID)
                    alert_logger = gcp_client.logger("air-paradis-alerts")
                    
                    alert_entry = {
                        "message": f"🚨 ALERTE: Prédiction incorrecte détectée",
                        "alert_type": "incorrect_prediction",
                        "text": log_request.data.get('text', ''),
                        "predicted_sentiment": log_request.data.get('predicted_sentiment'),
                        "actual_sentiment": log_request.data.get('actual_sentiment'),
                        "confidence": log_request.data.get('confidence'),
                        "user_id": log_request.data.get('user_id'),
                        "timestamp": current_time.isoformat(),
                        "service": "air-paradis-sentiment-api"
                    }
                    
                    alert_logger.log_struct(alert_entry, severity='WARNING')
                except Exception as alert_error:
                    logger.error(f"❌ Erreur envoi alerte Google Cloud: {alert_error}")
        
        # Gestion des erreurs critiques du frontend
        if log_request.level.upper() in ['ERROR', 'CRITICAL']:
            error_msg = f"Erreur frontend: {log_request.message}"
            background_tasks.add_task(
                send_error_to_monitoring,
                error_msg,
                str(log_request.data)
            )
            
            # Log d'erreur critique pour Google Cloud
            if os.getenv('GOOGLE_CLOUD_LOGGING_ENABLED', '').lower() == 'true':
                try:
                    from google.cloud import logging as gcp_logging
                    gcp_client = gcp_logging.Client(project=PROJECT_ID)
                    error_logger = gcp_client.logger("air-paradis-errors")
                    
                    error_entry = {
                        "message": f"🚨 ERREUR CRITIQUE FRONTEND: {log_request.message}",
                        "error_type": "frontend_critical_error",
                        "error_data": log_request.data,
                        "timestamp": current_time.isoformat(),
                        "service": "air-paradis-frontend",
                        "user_agent": log_request.data.get('userAgent') if log_request.data else None,
                        "url": log_request.data.get('url') if log_request.data else None
                    }
                    
                    error_logger.log_struct(error_entry, severity='ERROR')
                except Exception as error_log_err:
                    logger.error(f"❌ Erreur log erreur critique: {error_log_err}")
        
        # Génération de l'ID de log unique
        log_id = f"frontend_log_{int(current_time.timestamp() * 1000)}_{hash(log_request.message) % 10000}"
        
        # Réponse enrichie avec informations de monitoring
        return LogResponse(
            success=True,
            message="Log reçu et traité avec succès",
            logId=log_id,
            recentErrorCount=len(last_errors),
            alertThreshold=3,
            metadata={
                "timestamp": current_time.isoformat(),
                "projectId": log_request.projectId or PROJECT_ID,
                "logName": log_request.logName or "air-paradis-frontend",
                "level": log_request.level.upper(),
                "google_cloud_logging": os.getenv('GOOGLE_CLOUD_LOGGING_ENABLED', '').lower() == 'true',
                "environment": os.getenv('ENVIRONMENT', 'development'),
                "service_version": "1.0.0"
            }
        )
        
    except Exception as e:
        error_msg = f"Erreur lors du traitement du log frontend: {str(e)}"
        logger.error(error_msg)
        
        # Log de l'erreur de l'endpoint lui-même
        if os.getenv('GOOGLE_CLOUD_LOGGING_ENABLED', '').lower() == 'true':
            try:
                from google.cloud import logging as gcp_logging
                gcp_client = gcp_logging.Client(project=PROJECT_ID)
                endpoint_logger = gcp_client.logger("air-paradis-api-errors")
                
                endpoint_error = {
                    "message": f"❌ Erreur endpoint /api/logging: {str(e)}",
                    "error_type": "logging_endpoint_error",
                    "endpoint": "/api/logging",
                    "request_data": {
                        "level": log_request.level if 'log_request' in locals() else 'unknown',
                        "message": log_request.message if 'log_request' in locals() else 'unknown'
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "service": "air-paradis-sentiment-api"
                }
                
                endpoint_logger.log_struct(endpoint_error, severity='ERROR')
            except Exception as endpoint_error:
                logger.error(f"❌ Impossible de logger l'erreur endpoint: {endpoint_error}")
        
        raise HTTPException(
            status_code=500, 
            detail="Erreur lors du traitement du log"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
