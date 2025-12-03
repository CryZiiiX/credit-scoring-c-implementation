# Image de base officielle avec GCC
FROM gcc:12-bookworm

# Éviter les prompts interactifs lors de l'installation
ENV DEBIAN_FRONTEND=noninteractive

# Installer Python 3, pip et outils de build
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    make \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier requirements.txt et installer les dépendances Python
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copier tout le projet dans le container
COPY . .

# Compiler le projet C
RUN make clean && make

# Commande par défaut : exécuter le programme C
CMD ["./build/credit_risk_predictor"]

