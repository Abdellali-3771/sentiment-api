# ============================================
# 1️⃣ Base image : Python + TensorFlow
# ============================================
FROM python:3.12.3-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8080 \
    ENVIRONMENT=production \
    GOOGLE_CLOUD_LOGGING_ENABLED=true

# ============================================
# 2️⃣ Installation des dépendances système
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 3️⃣ Installation Python
# ============================================
WORKDIR /app

COPY requirements.txt .

# ✅ Pip à jour et installation des libs
RUN pip install --upgrade pip
RUN pip install --no-cache-dir tensorflow==2.16.1
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# 4️⃣ Copier le code source + modèles
# ============================================
# Copie ton code FastAPI et ton modèle
COPY app ./app
COPY models ./models

# Vérification du contenu du dossier models
RUN echo "🔍 Vérification du contenu du dossier models :" && \
    ls -lh ./models && \
    echo "📊 Taille totale des modèles :" && \
    du -sh ./models

# ============================================
# 5️⃣ Créer un utilisateur non-root (sécurité)
# ============================================
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# ============================================
# 6️⃣ Exposer le port pour Cloud Run
# ============================================
EXPOSE 8080

# ============================================
# 7️⃣ Health check (optionnel mais recommandé)
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ============================================
# 8️⃣ Commande de démarrage avec timeouts étendus
# ============================================
# IMPORTANT: Utiliser exec form sans backslash pour éviter les erreurs
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 300 --workers 1 --log-level info"]