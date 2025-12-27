from __future__ import annotations
import random
from typing import List, Tuple

from .representation import Chromosome

# ---------- SELECTION (2) ----------

def tournament_selection(population: List[Chromosome], fitnesses: List[float], k: int = 3) -> Chromosome:
    """Minimización: devuelve copia del mejor de k individuos."""
    idxs = random.sample(range(len(population)), k)
    best_i = min(idxs, key=lambda i: fitnesses[i])
    return population[best_i][:]

def roulette_selection(population: List[Chromosome], fitnesses: List[float], eps: float = 1e-9) -> Chromosome:
    """
    Ruleta adaptada a minimización: prob ~ 1/(fitness + eps).
    Ojo: si hay fitness muy pequeños, domina mucho. eps evita división por 0.
    """
    weights = [1.0 / (f + eps) for f in fitnesses]
    total = sum(weights)
    r = random.random() * total
    acc = 0.0
    for ch, w in zip(population, weights):
        acc += w
        if acc >= r:
            return ch[:]
    return population[-1][:]


# ---------- CROSSOVER (2) ----------

def one_point_crossover(a: Chromosome, b: Chromosome, p: float = 0.9) -> Tuple[Chromosome, Chromosome]:
    """Cruce de 1 punto."""
    if random.random() > p or len(a) < 2:
        return a[:], b[:]
    cut = random.randint(1, len(a) - 1)
    return a[:cut] + b[cut:], b[:cut] + a[cut:]

def uniform_crossover(a: Chromosome, b: Chromosome, p: float = 0.9) -> Tuple[Chromosome, Chromosome]:
    """Cruce uniforme: para cada gen, se decide de qué padre se hereda."""
    if random.random() > p:
        return a[:], b[:]
    c1, c2 = a[:], b[:]
    for i in range(len(a)):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2


# ---------- MUTATION (2) ----------

def random_reset_mutation(ch: Chromosome, n_colors: int, p_gene: float = 0.02) -> Chromosome:
    """Cada gen muta con prob p_gene reasignando un color aleatorio."""
    out = ch[:]
    for i in range(len(out)):
        if random.random() < p_gene:
            out[i] = random.randrange(n_colors)
    return out

def swap_mutation(ch: Chromosome, p: float = 0.2) -> Chromosome:
    """
    Intercambia los colores de dos posiciones al azar con prob p.
    Suele ayudar a explorar sin “romper” demasiado la estructura.
    """
    out = ch[:]
    if len(out) < 2 or random.random() > p:
        return out
    i, j = random.sample(range(len(out)), 2)
    out[i], out[j] = out[j], out[i]
    return out
