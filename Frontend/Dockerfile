# =========================================
# FRONTEND DEPLOY POUR CLOUD RUN (Nginx)
# =========================================
FROM nginx:stable-alpine

# Supprime la conf par défaut
RUN rm /etc/nginx/conf.d/default.conf

# Ajoute notre propre configuration avec le port 8080
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copie les fichiers du frontend (index.html, css, js)
COPY . /usr/share/nginx/html

# Expose le port requis par Cloud Run
EXPOSE 8080

# Lance Nginx
CMD ["nginx", "-g", "daemon off;"]
