# main.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from config import *
from genetic import Individual
from model import DynamicTransformer

# --- FITNESS FUNCTION ---
def evaluate_individual(ind):
    """Builds, trains, and evaluates an individual."""
    print(f"   > Evaluation: {ind.genes}...", end="")
    
    # Model Construction
    # e.g. : Input dim 10, Seq len 20, Output 2 classes
    model = DynamicTransformer(input_dim=10, output_dim=2, config_genes=ind.genes).to(DEVICE)
    
    # Length Calculation
    ind.n_params = model.count_parameters()
    if ind.n_params > MAX_PARAMS:
        print(f"Too large ({ind.n_params/1e6:.2f}M params)")
        ind.accuracy = 0.0
        ind.fitness = -1.0 # Penalty for too large models - model rejected
        return

    # Training Loop (Simplified for demonstration)
    # We are using a fake dataset here for illustration purposes
    inputs = torch.randn(BATCH_SIZE, 20, 10).to(DEVICE) # Batch, Seq, Dim
    targets = torch.randint(0, 2, (BATCH_SIZE,)).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(EPOCHS_PER_EVAL):
        optimizer.zero_grad() # Reset gradients
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, targets)
        loss.backward() # Backpropagation
        optimizer.step() # Update weights
        
    # Accuracy (Here we use the inverse Loss as a simplified accuracy for the example)
    ind.accuracy = 1.0 / (loss.item() + 1e-5) # Lower loss is better
    
    # Fitness = Accuracy weighted by size

    alpha = 0.1 
    size_ratio = ind.n_params / MAX_PARAMS

    ind.fitness = ind.accuracy - (alpha * size_ratio)
    
    print(f"Score: {ind.fitness:.4f} | Params: {ind.n_params}")
    score.append(ind.fitness)

# --- MAIN LOOP ---

score = [] # Store all scores for analysis
def main():
    print(f"Launching Genetic Algorithm for {NUM_GENERATIONS} generations...")
    
    # Initialization
    population = [Individual() for _ in range(POPULATION_SIZE)]
    
    for gen in range(NUM_GENERATIONS):
        print(f"\n--- GENERATION {gen + 1}/{NUM_GENERATIONS} ---")
        
        # Evaluation
        for ind in population:
            evaluate_individual(ind)
            
        # Sorting (Best first)
        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        print(f"🏆 Best of Gen {gen+1}: {best.genes} (Fit: {best.fitness:.4f})")
        
        # If it's the last generation, stop here
        if gen == NUM_GENERATIONS - 1:
            break
            
        # 4. SSelection & Reproduction
        next_gen = [best] # Elitism
        
        while len(next_gen) < POPULATION_SIZE:

            
            def tournament_selection(pop, k):
                # Select k individuals randomly from the population
                candidates = random.choices(pop, k=k) 
                # Return the best candidate (the one with the highest fitness)
                return max(candidates, key=lambda x: x.fitness)
            
            # Simple tournament
            parent1 = tournament_selection(population, TOURNAMENT_SIZE) # SSelect first parent
            parent2 = tournament_selection(population, TOURNAMENT_SIZE)
            
            # Crossover
            child = Individual.crossover(parent1, parent2)
            
            # Mutation
            child.mutate()
            next_gen.append(child)
            
        population = next_gen

    print("\nSEARCH COMPLETED.")
    print(f"Winning architecture: {population[0].genes}")
    print(f"Parameters: {population[0].n_params}")

    # Display a line plot of scores over generations
    try:
        import matplotlib.pyplot as plt

        plt.plot(score)
        plt.title("Scores of individuals over evaluations")
        plt.xlabel("Evaluations")
        plt.ylabel("Fitness score")
        plt.show()
    except ImportError:
        print("matplotlib is not installed, the score plot cannot be displayed.")

if __name__ == "__main__":
    main()