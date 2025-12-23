from pathlib import Path
from datasets import load_dimacs_col

def main():
    g = load_dimacs_col(Path("data/raw/queen7_7.col"))
    print("Vertices:", g.n_vertices)
    print("Edges:", len(g.edges))

if __name__ == "__main__":
    main()
