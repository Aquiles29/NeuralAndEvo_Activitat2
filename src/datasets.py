from pathlib import Path
from typing import List, Tuple
from .graph import Graph

def load_dimacs_col(path: str | Path) -> Graph:
    path = Path(path)
    n_vertices = None
    edges: List[Tuple[int, int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue

            parts = line.split()
            if parts[0] == "p":
                # p edge N M
                n_vertices = int(parts[2])
            elif parts[0] == "e":
                # e u v (1-indexed in file)
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                if u != v:
                    if u > v:
                        u, v = v, u
                    edges.append((u, v))

    if n_vertices is None:
        raise ValueError(f"No se encontró la línea 'p edge' en: {path}")

    edges = sorted(set(edges))
    return Graph(n_vertices=n_vertices, edges=edges)
