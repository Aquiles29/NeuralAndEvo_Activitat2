from pathlib import Path

from .datasets import load_dimacs_col
from .ga.representation import random_chromosome
from .ga.fitness import evaluate, fitness

def main():
    g = load_dimacs_col(Path("data/raw/queen7_7.col"))

    # por ahora elegimos un número de colores "de arranque"
    # (luego intentaremos bajarlo)
    n_colors = 10

    ch = random_chromosome(g.n_vertices, n_colors, seed=0)
    ev = evaluate(g, ch)
    f = fitness(g, ch)

    print("Dataset:", "queen7_7.col")
    print("Vertices:", g.n_vertices)
    print("Edges:", len(g.edges))
    print("n_colors (allowed):", n_colors)
    print("Colors used:", ev.colors_used)
    print("Conflicts:", ev.conflicts)
    print("Fitness:", f)

if __name__ == "__main__":
    main()
