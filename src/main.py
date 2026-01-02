from pathlib import Path

from .datasets import load_dimacs_col
from .ga.genetic_algorithm import run_ga, GAParams
from .ga.operators import roulette_selection, uniform_crossover, swap_mutation

def main():
    dataset = Path("data/raw/queen7_7.col")
    g = load_dimacs_col(dataset)

    n_colors = 9
    params = GAParams(
        population_size=300,
        generations=1500,
        elitism=2,
        seed=0,
        patience=200,
    )

    res = run_ga(
        graph=g,
        n_colors=n_colors,
        select_fn=lambda pop, fits: roulette_selection(pop, fits),
        crossover_fn=lambda a, b: uniform_crossover(a, b, p=0.9),
        mutate_fn=lambda ch: swap_mutation(ch, p=0.3),
        params=params,
        w_conflict=1000.0,
        w_colors=1.0,
    )

    print("Dataset:", dataset.name)
    print("V:", g.n_vertices, "E:", len(g.edges))
    print("k:", n_colors)
    print("best_conflicts:", res["best_conflicts"])
    print("best_colors_used:", res["best_colors_used"])
    print("best_fitness:", res["best_fitness"])
    print("stopped_generation:", res["stopped_generation"])

if __name__ == "__main__":
    main()
