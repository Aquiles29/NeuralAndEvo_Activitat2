from __future__ import annotations
import random
from typing import List, Tuple

from .representation import Chromosome

def tournament_selection(population: List[Chromosome], fitnesses: List[float], k: int = 3) -> Chromosome:
    """Minimización: devuelve una copia del mejor de k individuos aleatorios."""
    idxs = random.sample(range(len(population)), k)
    best_i = min(idxs, key=lambda i: fitnesses[i])
    return population[best_i][:]

def one_point_crossover(a: Chromosome, b: Chromosome, p: float = 0.9) -> Tuple[Chromosome, Chromosome]:
    """Cruce de 1 punto. Con prob (1-p) no cruza."""
    if random.random() > p or len(a) < 2:
        return a[:], b[:]
    cut = random.randint(1, len(a) - 1)
    return a[:cut] + b[cut:], b[:cut] + a[cut:]

def random_reset_mutation(ch: Chromosome, n_colors: int, p_gene: float = 0.02) -> Chromosome:
    """Cada gen muta con prob p_gene reasignando un color aleatorio [0, n_colors)."""
    out = ch[:]
    for i in range(len(out)):
        if random.random() < p_gene:
            out[i] = random.randrange(n_colors)
    return out
