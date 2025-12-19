# genetic.py
import random
from config import SEARCH_SPACE, MUTATION_RATE

class Individual:
    def __init__(self, genes=None):
        self.genes = genes if genes else self._random_genes()
        self.fitness = 0.0
        self.accuracy = 0.0
        self.n_params = 0

    def _random_genes(self):
        """Generate random genes."""
        genes = {}
        for key, choices in SEARCH_SPACE.items():
            genes[key] = random.choice(choices)
        self._fix_constraints(genes)
        return genes

    def _fix_constraints(self, genes):
        """Ensure d_model is divisible by n_heads (required for PyTorch)."""
        while genes['d_model'] % genes['n_heads'] != 0:
            # Either change the heads or the model. Here we change the heads.
            genes['n_heads'] = random.choice(SEARCH_SPACE['n_heads'])

    def mutate(self):
        """Randomly change a gene."""
        if random.random() < MUTATION_RATE:
            gene_key = random.choice(list(SEARCH_SPACE.keys()))
            self.genes[gene_key] = random.choice(SEARCH_SPACE[gene_key])
            self._fix_constraints(self.genes) # Fix if divisibility is broken
            
    @staticmethod
    def crossover(parent1, parent2):
        """Mix two parents."""
        child_genes = {}
        for key in SEARCH_SPACE.keys():
            # 50% chance to take the gene from the father or the mother
            child_genes[key] = random.choice([parent1.genes[key], parent2.genes[key]])
        
        child = Individual(genes=child_genes)
        child._fix_constraints(child.genes)
        return child