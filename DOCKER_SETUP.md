# Configuration Docker pour Ubuntu 22.04

## Installation de Docker

Si Docker n'est pas encore installé sur votre système, suivez ces étapes :

### 1. Installer Docker

```bash
# Mettre à jour les paquets
sudo apt update

# Installer Docker
sudo apt install -y docker.io

# Démarrer et activer Docker
sudo systemctl start docker
sudo systemctl enable docker
```

### 2. Configurer les permissions (optionnel mais recommandé)

Pour éviter d'utiliser `sudo` à chaque commande Docker :

```bash
# Ajouter votre utilisateur au groupe docker
sudo usermod -aG docker $USER

# Appliquer les changements (ou se déconnecter/reconnecter)
newgrp docker
```

### 3. Vérifier l'installation

```bash
# Vérifier la version de Docker
docker --version

# Tester avec l'image hello-world
docker run hello-world
```

## Utilisation avec le Projet

Une fois Docker installé et configuré :

### Build de l'image

```bash
# Se placer dans le répertoire du projet
cd "/home/maxiiimax/Bureau/FAC/M1/SEMESTRE-1/Techniques d'apprentissage artificiel/PROJET/PROGRAMME"

# Build l'image
docker build -t credit-risk-predictor:latest .
```

Le build prend environ 3-5 minutes la première fois (téléchargement de l'image de base + installation des dépendances).

### Exécution

```bash
# Exécution simple du programme C
docker run --rm credit-risk-predictor:latest

# Exécution avec sauvegarde des résultats localement
docker run --rm \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  credit-risk-predictor:latest

# Exécuter les tests
docker run --rm credit-risk-predictor:latest bash -c "cd tests && ./run_tests.sh"

# Comparaison avec scikit-learn
docker run --rm credit-risk-predictor:latest python3 scripts/compare_with_sklearn.py

# Mode interactif pour explorer
docker run --rm -it credit-risk-predictor:latest bash
```

## Dépannage

### Erreur "Permission denied"

Si vous obtenez une erreur de permission lors de l'utilisation de Docker :

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Nettoyage de l'espace disque

Si Docker occupe trop d'espace :

```bash
# Supprimer les images et containers inutilisés
docker system prune -a

# Vérifier l'espace disque disponible
df -h
```

### Rebuild complet

Si vous modifiez le code et souhaitez reconstruire l'image :

```bash
# Supprimer l'ancienne image
docker rmi credit-risk-predictor:latest

# Rebuild
docker build -t credit-risk-predictor:latest .
```

## Versions et Compatibilité

Pour connaître les versions exactes de tous les composants (GCC 12.5.0, Python 3.11.2, bibliothèques scientifiques), consultez le fichier **`VERSIONS.md`** qui documente en détail :
- Les versions de tous les composants
- La matrice de compatibilité
- Les commandes de vérification
- Les notes techniques

## Pour le Correcteur/Professeur

Si vous êtes un correcteur et souhaitez tester le projet :

```bash
# 1. Extraire l'archive ou cloner le dépôt
cd credit-risk-project

# 2. Build l'image (une seule fois)
docker build -t credit-risk:latest .

# 3. Exécuter le programme
docker run --rm credit-risk:latest

# 4. (Optionnel) Exécuter les tests
docker run --rm credit-risk:latest bash -c "cd tests && ./run_tests.sh"
```

Temps total estimé : ~5 minutes (3 min build + 2 min exécution).

## Résultats Attendus

Lors de l'exécution du programme, vous devriez voir :

- Compilation sans erreurs ni warnings
- Chargement du dataset (32 581 lignes)
- Entraînement de la Régression Logistique (~0.4s)
- Entraînement de l'Arbre de Décision (~0.3s)
- Affichage des métriques :
  - Logistic Regression : Accuracy ~80.7%, AUC-ROC ~81.8%
  - Decision Tree : Accuracy ~92.7%, AUC-ROC ~90.4%
- Comparaison des deux modèles

## Avantages de cette Approche

- **Reproductibilité** : Même environnement sur toutes les machines
- **Isolation** : Pas d'interférence avec les installations système
- **Simplicité** : Une seule commande pour exécuter tout le projet
- **Professionnalisme** : Démontre la maîtrise des outils modernes de développement

