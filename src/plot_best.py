from __future__ import annotations
from pathlib import Path
import argparse

import matplotlib.pyplot as plt

from .datasets import load_dimacs_col
from .ga.genetic_algorithm import run_ga, GAParams
from .ga.operators import (
    tournament_selection,
    roulette_selection,
    one_point_crossover,
    uniform_crossover,
    random_reset_mutation,
    swap_mutation,
)

def get_pipeline(config: str, k: int):
    """
    Devuelve (select_fn, crossover_fn, mutate_fn) según el nombre de config.
    """
    config = config.lower().strip()

    if config == "tour_1pt_reset":
        sel = lambda pop, fits: tournament_selection(pop, fits, k=3)
        cross = lambda a, b: one_point_crossover(a, b, p=0.9)
        mut = lambda ch: random_reset_mutation(ch, n_colors=k, p_gene=0.02)
        return sel, cross, mut

    if config == "tour_uniform_reset":
        sel = lambda pop, fits: tournament_selection(pop, fits, k=3)
        cross = lambda a, b: uniform_crossover(a, b, p=0.9)
        mut = lambda ch: random_reset_mutation(ch, n_colors=k, p_gene=0.02)
        return sel, cross, mut

    if config == "tour_1pt_swap":
        sel = lambda pop, fits: tournament_selection(pop, fits, k=3)
        cross = lambda a, b: one_point_crossover(a, b, p=0.9)
        mut = lambda ch: swap_mutation(ch, p=0.3)
        return sel, cross, mut

    if config == "roulette_1pt_reset":
        sel = lambda pop, fits: roulette_selection(pop, fits)
        cross = lambda a, b: one_point_crossover(a, b, p=0.9)
        mut = lambda ch: random_reset_mutation(ch, n_colors=k, p_gene=0.02)
        return sel, cross, mut

    if config == "roulette_uniform_swap":
        sel = lambda pop, fits: roulette_selection(pop, fits)
        cross = lambda a, b: uniform_crossover(a, b, p=0.9)
        mut = lambda ch: swap_mutation(ch, p=0.3)
        return sel, cross, mut

    if config == "roulette_uniform_reset":
        sel = lambda pop, fits: roulette_selection(pop, fits)
        cross = lambda a, b: uniform_crossover(a, b, p=0.9)
        mut = lambda ch: random_reset_mutation(ch, n_colors=k, p_gene=0.02)
        return sel, cross, mut

    raise ValueError(
        "Config desconocida. Usa una de: "
        "tour_1pt_reset, tour_uniform_reset, tour_1pt_swap, "
        "roulette_1pt_reset, roulette_uniform_swap, roulette_uniform_reset"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Ruta del dataset .col (ej: data/raw/queen7_7.col)")
    ap.add_argument("--k", type=int, required=True, help="Numero de colores permitido (k)")
    ap.add_argument("--config", required=True, help="Config pipeline (ej: roulette_uniform_swap)")
    ap.add_argument("--out", default="", help="Ruta salida PNG (opcional). Si no, se autogenera en results/")
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--gen", type=int, default=1500)
    ap.add_argument("--elitism", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=200)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    k = args.k
    config = args.config

    g = load_dimacs_col(dataset_path)

    params = GAParams(
        population_size=args.pop,
        generations=args.gen,
        elitism=args.elitism,
        seed=args.seed,
        patience=args.patience,
    )

    sel, cross, mut = get_pipeline(config, k)

    res = run_ga(
        graph=g,
        n_colors=k,
        select_fn=sel,
        crossover_fn=cross,
        mutate_fn=mut,
        params=params,
        w_conflict=1000.0,
        w_colors=1.0,
    )

    history = res["history_best"]

    # output path
    if args.out:
        out_path = Path(args.out)
    else:
        safe_name = dataset_path.stem.replace(".", "_")
        out_path = Path(f"results/fitness_{safe_name}_k{k}_{config}.png")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # plot
    plt.figure()
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness so far")
    plt.title(f"{dataset_path.name} | k={k} | {config}\n"
              f"best_conflicts={res['best_conflicts']}, best_colors_used={res['best_colors_used']}")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print("Dataset:", dataset_path.name)
    print("V:", g.n_vertices, "E:", len(g.edges))
    print("k:", k, "config:", config)
    print("best_conflicts:", res["best_conflicts"])
    print("best_colors_used:", res["best_colors_used"])
    print("best_fitness:", res["best_fitness"])
    print("stopped_generation:", res["stopped_generation"])
    print("Saved plot:", out_path)

if __name__ == "__main__":
    main()
