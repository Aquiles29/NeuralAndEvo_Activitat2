from pathlib import Path

from .datasets import load_dimacs_col
from .ga.genetic_algorithm import run_ga, GAParams
from .ga.operators import tournament_selection, one_point_crossover, random_reset_mutation

def main():
    dataset = Path("data/raw/queen7_7.col")
    g = load_dimacs_col(dataset)

    n_colors = 9  # luego lo intentaremos bajar
    params = GAParams(population_size=300, generations=1500, elitism=2, seed=0)

    mutate = lambda ch: random_reset_mutation(ch, n_colors=n_colors, p_gene=0.02)

    result = run_ga(
        graph=g,
        n_colors=n_colors,
        select_fn=lambda pop, fits: tournament_selection(pop, fits, k=3),
        crossover_fn=lambda a, b: one_point_crossover(a, b, p=0.9),
        mutate_fn=mutate,
        params=params,
        w_conflict=1000.0,
        w_colors=1.0,
    )

    print("Dataset:", dataset.name)
    print("Vertices:", g.n_vertices)
    print("Edges:", len(g.edges))
    print("Allowed colors:", n_colors)
    print("Best conflicts:", result["best_conflicts"])
    print("Best colors used:", result["best_colors_used"])
    print("Best fitness:", result["best_fitness"])
    print("Best fitness (last gen):", result["history_best"][-1])

if __name__ == "__main__":
    main()
