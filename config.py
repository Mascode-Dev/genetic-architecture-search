# config.py

# --- ESPACE DE RECHERCHE (Gènes possibles) ---
# Le modèle piochera une valeur dans chaque liste
SEARCH_SPACE = {
    "n_layers": [1, 2, 4, 6],           # Profondeur du réseau
    "n_heads": [2, 4, 8],               # Têtes d'attention
    "d_model": [64, 128, 256],          # Taille des vecteurs (largeur)
    "dim_feedforward_ratio": [1, 2, 4]  # Facteur d'expansion du MLP
}

# --- PARAMÈTRES GÉNÉTIQUES ---
POPULATION_SIZE = 10        # Nombre de modèles par génération
NUM_GENERATIONS = 10         # Nombre de cycles d'évolution
MUTATION_RATE = 0.2         # 20% de chance de mutation
TOURNAMENT_SIZE = 3         # Pour la sélection des parents

# --- CONTRAINTES & ENTRAÎNEMENT ---
MAX_PARAMS = 1_000_000      # Limite stricte : 1 Million de paramètres
DEVICE = "cuda"              # Mettre "cuda" si tu as un GPU NVIDIA
EPOCHS_PER_EVAL = 15
BATCH_SIZE = 32