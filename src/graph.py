from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Graph:
    n_vertices: int
    edges: List[Tuple[int, int]]  # 0-indexed
