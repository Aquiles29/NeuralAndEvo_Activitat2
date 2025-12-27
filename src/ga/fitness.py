from __future__ import annotations
from dataclasses import dataclass
from typing import Set

from ..graph import Graph
from .representation import Chromosome

@dataclass(frozen=True)
class Evaluation:
    conflicts: int        # number of edges with same-color endpoints
    colors_used: int      # number of distinct colors in chromosome

def evaluate(graph: Graph, ch: Chromosome) -> Evaluation:
    conflicts = 0
    for u, v in graph.edges:
        if ch[u] == ch[v]:
            conflicts += 1
    colors_used = len(set(ch))
    return Evaluation(conflicts=conflicts, colors_used=colors_used)

def fitness(graph: Graph, ch: Chromosome, w_conflict: float = 1000.0, w_colors: float = 1.0) -> float:
    """
    Minimization objective:
      - heavily penalize conflicts (hard constraint)
      - then minimize number of colors used (soft objective once conflicts are 0)
    """
    ev = evaluate(graph, ch)
    return w_conflict * ev.conflicts + w_colors * ev.colors_used
