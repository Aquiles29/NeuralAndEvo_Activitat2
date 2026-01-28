from __future__ import annotations
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from .datasets import load_dimacs_col
from .ga.genetic_algorithm import run_ga, GAParams
from .ga.operators import (
    tournament_selection,
    roulette_selection_rank,  # CHANGED
    one_point_crossover,
    uniform_crossover,
    random_reset_mutation,
    swap_mutation,
)

def get_pipeline(config: str, k: int):
    config = config.lower().strip()

    if config == "tour_1pt_reset":
        sel = lambda pop, scores: tournament_selection(pop, scores, k=3)  # CHANGED
        cross = lambda a, b: one_point_crossover(a, b, p=0.9)
        mut = lambda ch: random_reset_mutation(ch, n_colors=k, p_gene=0.02)
        return sel, cross, mut

    if config == "roulette_rank_uniform_swap":
        sel = lambda pop, scores: roulette_selection_rank(pop, scores)    # CHANGED
        cross = lambda a, b: uniform_crossover(a, b, p=0.9)
        mut = lambda ch: swap_mutation(ch, p=0.3)
        return sel, cross, mut

    raise ValueError("Config desconocida para este ejemplo.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="")
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
        penalty=1000,
    )

    # CHANGED: use scalar history for a single curve
    history = res["history_best_scalar"]

    if args.out:
        out_path = Path(args.out)
    else:
        safe = dataset_path.stem.replace(".", "_")
        out_path = Path(f"results/fitness_{safe}_k{k}_{config}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (scalar, for plotting)")
    plt.title(f"{dataset_path.name} | k={k} | {config}\n"
              f"best_score={res['best_score']}")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print("Saved plot:", out_path)
    print("best_score:", res["best_score"])
    print("best_conflicts:", res["best_conflicts"])
    print("best_colors_used:", res["best_colors_used"])
    print("stopped_generation:", res["stopped_generation"])

if __name__ == "__main__":
    main()
