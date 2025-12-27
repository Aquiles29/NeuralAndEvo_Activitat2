from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Tuple

from ..graph import Graph
from .representation import Chromosome, random_chromosome
from .fitness import fitness, evaluate

SelectFn = Callable[[List[Chromosome], List[float]], Chromosome]
CrossoverFn = Callable[[Chromosome, Chromosome], Tuple[Chromosome, Chromosome]]
MutateFn = Callable[[Chromosome], Chromosome]

@dataclass
class GAParams:
    population_size: int = 200
    generations: int = 300
    elitism: int = 2
    seed: int | None = 0

def run_ga(
    graph: Graph,
    n_colors: int,
    select_fn: SelectFn,
    crossover_fn: CrossoverFn,
    mutate_fn: MutateFn,
    params: GAParams,
    w_conflict: float = 1000.0,
    w_colors: float = 1.0,
) -> Dict[str, Any]:
    if params.seed is not None:
        random.seed(params.seed)

    # init population
    population: List[Chromosome] = [
        random_chromosome(graph.n_vertices, n_colors) for _ in range(params.population_size)
    ]

    best_ch: Chromosome | None = None
    best_fit = float("inf")
    history_best: List[float] = []

    for _gen in range(params.generations):
        fits = [fitness(graph, ch, w_conflict=w_conflict, w_colors=w_colors) for ch in population]

        # update global best
        best_idx = min(range(len(population)), key=lambda i: fits[i])
        if fits[best_idx] < best_fit:
            best_fit = fits[best_idx]
            best_ch = population[best_idx][:]

        history_best.append(best_fit)

        # elitism: carry best N from current population
        elite_idxs = sorted(range(len(population)), key=lambda i: fits[i])[: params.elitism]
        new_pop: List[Chromosome] = [population[i][:] for i in elite_idxs]

        # rest by reproduction
        while len(new_pop) < params.population_size:
            p1 = select_fn(population, fits)
            p2 = select_fn(population, fits)
            c1, c2 = crossover_fn(p1, p2)
            c1 = mutate_fn(c1)
            c2 = mutate_fn(c2)

            new_pop.append(c1)
            if len(new_pop) < params.population_size:
                new_pop.append(c2)

        population = new_pop

    if best_ch is None:
        best_ch = population[0][:]

    best_eval = evaluate(graph, best_ch)
    return {
        "best_chromosome": best_ch,
        "best_fitness": best_fit,
        "best_conflicts": best_eval.conflicts,
        "best_colors_used": best_eval.colors_used,
        "history_best": history_best,
    }
