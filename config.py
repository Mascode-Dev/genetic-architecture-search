# config.py
# Search spaces (possible values for the genes)
SEARCH_SPACE = {
    "n_layers": [1, 2, 4, 6], # Networks Depth
    "n_heads": [2, 4, 8], # Attention heads
    "d_model": [64, 128, 256], # Vectors length
    "dim_feedforward_ratio": [1, 2, 4] # MLP Growing Factor
}

# Genetic Parameters
POPULATION_SIZE = 30  # Number of models per generation
NUM_GENERATIONS = 75  # Number of evolution cycles
MUTATION_RATE = 0.15  # Mutation probability
TOURNAMENT_SIZE = 4 # Size of tournament for parents selection

# Constaints & Training
MAX_PARAMS = 1_000_000 # Number of parameters - 1M by default to take a cheap cost model
DEVICE = "cuda" # Type of device GPU or CPU
EPOCHS_PER_EVAL = 10 # Number of iteration per evaluation
BATCH_SIZE = 32 # Batch size