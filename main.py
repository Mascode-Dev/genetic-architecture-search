# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from config import *
from genetic import Individual
from model import DynamicTransformer

# --- FONCTION D'ÉVALUATION (FITNESS) ---
def evaluate_individual(ind):
    """Construit, entraîne et évalue un individu."""
    print(f"   > Évaluation: {ind.genes}...", end="")
    
    # 1. Création du modèle
    # Pour l'exemple : Input dim 10, Seq len 20, Output 2 classes
    model = DynamicTransformer(input_dim=10, output_dim=2, config_genes=ind.genes).to(DEVICE)
    
    # 2. Vérification Taille
    ind.n_params = model.count_parameters()
    if ind.n_params > MAX_PARAMS:
        print(f" ❌ Trop gros ({ind.n_params/1e6:.2f}M params)")
        ind.accuracy = 0.0
        ind.fitness = -1.0 # Pénalité mortelle
        return

    # 3. Mini-Entraînement (Proxy Task)
    # Données bidons pour tester l'algo rapidement
    inputs = torch.randn(BATCH_SIZE, 20, 10).to(DEVICE) # Batch, Seq, Dim
    targets = torch.randint(0, 2, (BATCH_SIZE,)).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(EPOCHS_PER_EVAL): # Boucle d'entraînement courte
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    # 4. Score (Ici on utilise la Loss inverse comme précision simplifiée pour l'exemple)
    # Dans un vrai cas, on ferait un validation loop
    ind.accuracy = 1.0 / (loss.item() + 1e-5) # Plus loss est basse, mieux c'est
    
    # Fitness = Précision pondérée par la taille (optionnel)
    ind.fitness = ind.accuracy 
    
    print(f" ✅ Score: {ind.fitness:.4f} | Params: {ind.n_params}")

# --- BOUCLE PRINCIPALE ---
def main():
    print(f"🧬 Lancement de l'Algo Génétique sur {NUM_GENERATIONS} générations...")
    
    # 1. Initialisation
    population = [Individual() for _ in range(POPULATION_SIZE)]
    
    for gen in range(NUM_GENERATIONS):
        print(f"\n--- GÉNÉRATION {gen + 1}/{NUM_GENERATIONS} ---")
        
        # 2. Évaluation
        for ind in population:
            evaluate_individual(ind)
            
        # 3. Tri (Les meilleurs en premier)
        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        print(f"🏆 Meilleur de Gen {gen+1}: {best.genes} (Fit: {best.fitness:.4f})")
        
        # Si c'est la dernière génération, on s'arrête
        if gen == NUM_GENERATIONS - 1:
            break
            
        # 4. Sélection & Reproduction
        next_gen = [best] # Élitisme : on garde le meilleur tel quel
        
        while len(next_gen) < POPULATION_SIZE:
            # Tournoi simple
            parent1 = random.choice(population[:5]) # On prend parmi le top 50%
            parent2 = random.choice(population[:5])
            
            # Croisement
            child = Individual.crossover(parent1, parent2)
            
            # Mutation
            child.mutate()
            next_gen.append(child)
            
        population = next_gen

    print("\n🏁 RECHERCHE TERMINÉE.")
    print(f"Architecture gagnante : {population[0].genes}")
    print(f"Paramètres : {population[0].n_params}")

if __name__ == "__main__":
    main()