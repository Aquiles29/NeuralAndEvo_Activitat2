from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Tuple

from ..graph import Graph
from .representation import Chromosome, random_chromosome
from .fitness import evaluate, score, fitness_scalar, Evaluation  # CHANGED

SelectFn = Callable[[List[Chromosome], List[Tuple[int, int]]], Chromosome]  # CHANGED
CrossoverFn = Callable[[Chromosome, Chromosome], Tuple[Chromosome, Chromosome]]
MutateFn = Callable[[Chromosome], Chromosome]

@dataclass
class GAParams:
    population_size: int = 200
    generations: int = 300
    elitism: int = 2
    seed: int | None = 0
    patience: int = 200

def run_ga(
    graph: Graph,
    n_colors: int,
    select_fn: SelectFn,            # CHANGED: receives scores now
    crossover_fn: CrossoverFn,
    mutate_fn: MutateFn,
    params: GAParams,
    penalty: int = 1000,            # NEW: only for plotting scalar
) -> Dict[str, Any]:
    if params.seed is not None:
        random.seed(params.seed)

    # init population
    population: List[Chromosome] = [
        random_chromosome(graph.n_vertices, n_colors) for _ in range(params.population_size)
    ]

    # NEW: best is defined by lexicographic score
    best_ch: Chromosome | None = None
    best_ev: Evaluation | None = None
    best_score: Tuple[int, int] = (10**18, 10**18)  # large initial

    # NEW: history for plots (scalar) + history for analysis (score)
    history_best_scalar: List[float] = []
    history_best_score: List[Tuple[int, int]] = []

    # stationary tracking based on score (not scalar)
    no_improve = 0
    stopped_at = params.generations

    for gen in range(params.generations):
        # CHANGED: evaluate population explicitly
        evaluations = [evaluate(graph, ch) for ch in population]
        scores = [score(ev) for ev in evaluations]  # (conflicts, colors_used)

        # CHANGED: choose best in current population by lexicographic score
        best_idx = min(range(len(population)), key=lambda i: scores[i])
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_ch = population[best_idx][:]
            best_ev = evaluations[best_idx]
            no_improve = 0
        else:
            no_improve += 1

        # NEW: record history
        history_best_score.append(best_score)
        # scalar only for plot readability
        history_best_scalar.append(fitness_scalar(best_ev if best_ev is not None else evaluations[best_idx], penalty=penalty))

        # stationary stop
        if no_improve >= params.patience:
            stopped_at = gen
            break

        # CHANGED: elitism based on score (not scalar)
        elite_idxs = sorted(range(len(population)), key=lambda i: scores[i])[: params.elitism]
        new_pop: List[Chromosome] = [population[i][:] for i in elite_idxs]

        # reproduction
        while len(new_pop) < params.population_size:
            p1 = select_fn(population, scores)      # CHANGED: scores passed
            p2 = select_fn(population, scores)
            c1, c2 = crossover_fn(p1, p2)
            c1 = mutate_fn(c1)
            c2 = mutate_fn(c2)

            new_pop.append(c1)
            if len(new_pop) < params.population_size:
                new_pop.append(c2)

        population = new_pop

    if best_ch is None:
        best_ch = population[0][:]
        best_ev = evaluate(graph, best_ch)
        best_score = score(best_ev)

    assert best_ev is not None

    return {
        "best_chromosome": best_ch,
        # NEW: explicit "best" definition
        "best_score": best_score,  # (conflicts, colors_used)
        "best_conflicts": best_ev.conflicts,
        "best_colors_used": best_ev.colors_used,

        # NEW: histories
        "history_best_score": history_best_score,
        "history_best_scalar": history_best_scalar,

        "stopped_generation": stopped_at,
        "no_improve_generations": no_improve,
    }
