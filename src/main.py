# main.py

import networkx as nx
import os
from util.af_parser import parse_af_file
from semantics.dbs import Dbs

if __name__ == '__main__':
    print("--- Running Dbs test on a large AF from file ---")

    af_file_path = os.path.join("data/benchmarks2023/main", "admbuster_2500000.af")

    print("Load the graph using the parser...")
    arg_framework = parse_af_file(af_file_path)
    print(f"Graph loaded with {arg_framework.number_of_nodes()} nodes and {arg_framework.number_of_edges()} edges.")

    print("Calculate the ranking...")
    dbs = Dbs(arg_framework, max_path_length=20)

    # If you need to check, print only a small sample
    print("\n1. Discussion Vectors (Sample of first 10 arguments):")
    discussion_vectors = dbs.get_discussion_vectors()
    for i, arg in enumerate(sorted(discussion_vectors.keys())):
        if i >= 10:
            break
        vector = discussion_vectors[arg]
        print(f"   Dis({arg}) = {vector}")

    print("\n2. Final Ranking Order (most to least acceptable):")
    ranking = dbs.get_ranking()

    # --- AVOID PRINTING THE FULL RANKING STRING ON LARGE GRAPHS ---
    # This can also be slow and produce an unreadably long line.
    print(f"Ranking calculated with {len(ranking)} rank groups.")
    print("Top 5 rank groups:")
    for i, rank_group in enumerate(ranking[:5]):
        print(f"  Rank {i+1}: {sorted(list(rank_group)[:10])} (showing up to 10 args)")