from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

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

# NEW: explicit comparison key (what "best" means)
def score(ev: Evaluation) -> Tuple[int, int]:
    """
    Lexicographic minimization:
      1) minimize conflicts (feasibility)
      2) among equal conflicts, minimize colors_used
    """
    return (ev.conflicts, ev.colors_used)

# NEW: optional scalar ONLY for plotting (not for deciding best)
def fitness_scalar(ev: Evaluation, penalty: int = 1000) -> float:
    """
    Only used to plot a single curve.
    NOTE: The algorithm does NOT use this scalar to decide best solutions.
    """
    return float(ev.conflicts * penalty + ev.colors_used)
