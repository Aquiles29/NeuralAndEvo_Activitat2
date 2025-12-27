from __future__ import annotations
import random
from typing import List

Chromosome = List[int]  # gene i = color assigned to vertex i

def random_chromosome(n_vertices: int, n_colors: int, seed: int | None = None) -> Chromosome:
    if seed is not None:
        random.seed(seed)
    return [random.randrange(n_colors) for _ in range(n_vertices)]
