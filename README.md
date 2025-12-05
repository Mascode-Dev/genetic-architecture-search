# Genetic Architecture Search 🧬🤖

> **Neural Architecture Search (NAS) for Transformers using Genetic Algorithms.**

This project explores the use of **Evolutionary Algorithms** to automatically discover efficient Large Language Model (LLM) architectures. Instead of manually tuning hyperparameters, we use a Genetic Algorithm (GA) to find the optimal trade-off between **model accuracy** and **computational cost** (parameter count).

## 📝 Context
Modern LLMs are powerful but computationally expensive to train and deploy. Designing the "perfect" architecture—balancing depth, width, and attention heads—is a combinatorial optimization problem that is infeasible to solve via brute force.

This project implements a **Genetic Algorithm** to search the architectural space, aiming to evolve a Transformer model that performs well on a specific task while adhering to strict size constraints (e.g., < 5M parameters).

## 🎯 Objectives
- **Automate Design:** Implement a GA to evolve a population of Dynamic Transformers.
- **Constraint Optimization:** Maximize validation accuracy while minimizing parameter count.
- **Efficiency:** Demonstrate that evolutionary search finds better architectures than random selection.

## 🧠 Technical Approach

### 1. The Search Space (The Genome)
Each candidate architecture is represented by a genome `[L, H, D, F]`:
- **L (Layers)**: Depth of the network (e.g., 2, 4, 6).
- **H (Heads)**: Number of Attention Heads (e.g., 2, 4, 8).
- **D (Embedding Dim)**: Hidden dimension width (e.g., 128, 256).
- **F (Forward Expansion)**: Expansion ratio of the MLP feed-forward layer.

### 2. The Evolutionary Cycle


1.  **Initialization**: A population of random architectures is generated.
2.  **Evaluation (Fitness)**: Each model is built and trained for a short period (Proxy Task) to measure accuracy. A penalty is applied if the model exceeds the parameter budget.
    * $$Fitness = Accuracy - (\alpha \times SizePenalty)$$
3.  **Selection**: Tournament selection is used to pick the fittest parents.
4.  **Crossover**: Parents swap "genes" (hyperparameters) to create offspring.
5.  **Mutation**: Random changes are introduced (e.g., changing layer count) to maintain diversity.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Mascode-Dev/Evo-Transformer-NAS.git
cd Evo-Transformer-NAS

# Install dependencies
pip install torch torchvision numpy
