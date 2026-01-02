from __future__ import annotations
from pathlib import Path
import csv
from typing import Callable, Dict, Any, List, Tuple

from src.datasets import load_dimacs_col
from src.ga.genetic_algorithm import run_ga, GAParams
from src.ga.operators import (
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
    res_out = {
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
    return res_out

def main():
    dataset_path = Path("data/raw/queen7_7.col")

    # Cambia aquí el n_colors que quieras evaluar
    n_colors = 9

    params = GAParams(
        population_size=300,
        generations=1500,
        elitism=2,
        seed=0,
        patience=200,
    )

    configs: List[Tuple[str, Callable, Callable, Callable]] = [
        # 6 combinaciones (mínimas y defendibles)
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
    out_path = Path("results/experiments_queen7_7.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", out_path)

if __name__ == "__main__":
    main()
