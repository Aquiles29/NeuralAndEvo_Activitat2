from __future__ import annotations
import random
from typing import List, Tuple

from .representation import Chromosome

# ---------- SELECTION (2) ----------

# CHANGED: now receives scores (Tuple[int,int]) instead of fitness floats
def tournament_selection(population: List[Chromosome], scores: List[Tuple[int, int]], k: int = 3) -> Chromosome:
    """Minimization with lexicographic scores: returns a copy of the best of k."""
    idxs = random.sample(range(len(population)), k)
    best_i = min(idxs, key=lambda i: scores[i])
    return population[best_i][:]

# NEW/CHANGED: rank-based roulette for minimization (more robust than 1/(fitness))
def roulette_selection_rank(population: List[Chromosome], scores: List[Tuple[int, int]]) -> Chromosome:
    """
    Rank-based roulette selection (minimization).
    Best score gets highest probability, worst gets lowest.
    This avoids numerical instability and makes the intention clear.
    """
    order = sorted(range(len(population)), key=lambda i: scores[i])  # best -> worst
    n = len(population)
    # weights: n, n-1, ..., 1
    weights = [n - r for r in range(n)]
    total = sum(weights)
    r = random.randint(1, total)
    acc = 0
    for idx, w in zip(order, weights):
        acc += w
        if acc >= r:
            return population[idx][:]
    return population[order[-1]][:]


# ---------- CROSSOVER (2) ----------

def one_point_crossover(a: Chromosome, b: Chromosome, p: float = 0.9) -> Tuple[Chromosome, Chromosome]:
    if random.random() > p or len(a) < 2:
        return a[:], b[:]
    cut = random.randint(1, len(a) - 1)
    return a[:cut] + b[cut:], b[:cut] + a[cut:]

def uniform_crossover(a: Chromosome, b: Chromosome, p: float = 0.9) -> Tuple[Chromosome, Chromosome]:
    if random.random() > p:
        return a[:], b[:]
    c1, c2 = a[:], b[:]
    for i in range(len(a)):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2


# ---------- MUTATION (2) ----------

def random_reset_mutation(ch: Chromosome, n_colors: int, p_gene: float = 0.02) -> Chromosome:
    out = ch[:]
    for i in range(len(out)):
        if random.random() < p_gene:
            out[i] = random.randrange(n_colors)
    return out

def swap_mutation(ch: Chromosome, p: float = 0.2) -> Chromosome:
    out = ch[:]
    if len(out) < 2 or random.random() > p:
        return out
    i, j = random.sample(range(len(out)), 2)
    out[i], out[j] = out[j], out[i]
    return out
