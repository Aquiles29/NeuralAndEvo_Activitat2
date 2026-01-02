from __future__ import annotations
from pathlib import Path
import csv
import argparse
from typing import Callable, Dict, Any, List, Tuple

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

def run_one(name: str, dataset_path: Path, n_colors: int, params: GAParams,
            select_fn, crossover_fn, mutate_fn) -> Dict[str, Any]:
    g = load_dimacs_col(dataset_path)
    res = run_ga(
        graph=g,
        n_colors=n_colors,
        select_fn=select_fn,
        crossover_fn=crossover_fn,
        mutate_fn=mutate_fn,
        params=params,
        w_conflict=1000.0,
        w_colors=1.0,
    )
    return {
        "config": name,
        "dataset": dataset_path.name,
        "n_vertices": g.n_vertices,
        "n_edges": len(g.edges),
        "allowed_colors": n_colors,
        "best_conflicts": res["best_conflicts"],
        "best_colors_used": res["best_colors_used"],
        "best_fitness": res["best_fitness"],
        "stopped_generation": res["stopped_generation"],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Path to .col dataset (e.g., data/raw/queen7_7.col)")
    ap.add_argument("--k", type=int, required=True, help="Allowed number of colors")
    ap.add_argument("--out", type=str, default="", help="Optional output CSV path")
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--gen", type=int, default=1500)
    ap.add_argument("--elitism", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=200)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    n_colors = args.k

    params = GAParams(
        population_size=args.pop,
        generations=args.gen,
        elitism=args.elitism,
        seed=args.seed,
        patience=args.patience,
    )

    configs: List[Tuple[str, Callable, Callable, Callable]] = [
        ("tour + 1pt + reset",
         lambda pop, fits: tournament_selection(pop, fits, k=3),
         lambda a, b: one_point_crossover(a, b, p=0.9),
         lambda ch: random_reset_mutation(ch, n_colors=n_colors, p_gene=0.02)),

        ("tour + uniform + reset",
         lambda pop, fits: tournament_selection(pop, fits, k=3),
         lambda a, b: uniform_crossover(a, b, p=0.9),
         lambda ch: random_reset_mutation(ch, n_colors=n_colors, p_gene=0.02)),

        ("tour + 1pt + swap",
         lambda pop, fits: tournament_selection(pop, fits, k=3),
         lambda a, b: one_point_crossover(a, b, p=0.9),
         lambda ch: swap_mutation(ch, p=0.3)),

        ("roulette + 1pt + reset",
         lambda pop, fits: roulette_selection(pop, fits),
         lambda a, b: one_point_crossover(a, b, p=0.9),
         lambda ch: random_reset_mutation(ch, n_colors=n_colors, p_gene=0.02)),

        ("roulette + uniform + swap",
         lambda pop, fits: roulette_selection(pop, fits),
         lambda a, b: uniform_crossover(a, b, p=0.9),
         lambda ch: swap_mutation(ch, p=0.3)),

        ("roulette + uniform + reset",
         lambda pop, fits: roulette_selection(pop, fits),
         lambda a, b: uniform_crossover(a, b, p=0.9),
         lambda ch: random_reset_mutation(ch, n_colors=n_colors, p_gene=0.02)),
    ]

    rows: List[Dict[str, Any]] = []
    for name, sel, cross, mut in configs:
        rows.append(run_one(name, dataset_path, n_colors, params, sel, cross, mut))

    # print table
    print(f"Results for {dataset_path.name} (allowed_colors={n_colors})")
    print("-" * 95)
    header = ["config", "best_conflicts", "best_colors_used", "best_fitness", "stopped_generation"]
    print("{:<28} {:>14} {:>16} {:>12} {:>18}".format(*header))
    for r in rows:
        print("{:<28} {:>14} {:>16} {:>12} {:>18}".format(
            r["config"], r["best_conflicts"], r["best_colors_used"],
            f"{r['best_fitness']:.1f}", r["stopped_generation"]
        ))

    # save csv
    out_path = Path(args.out) if args.out else Path(f"results/experiments_{dataset_path.stem}_k{n_colors}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
