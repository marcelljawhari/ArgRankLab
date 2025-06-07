import networkx as nx

def parse_af_file(file_path: str) -> nx.DiGraph:
    """
    Parses an argumentation framework from a file in the ICCMA '.af' format.

    The function reads a file where arguments are indexed by positive integers.
    It handles the 'p af <n>' line to define arguments and subsequent lines
    to define attacks, ignoring comments.

    Args:
        file_path: The path to the .af file.

    Returns:
        A networkx.DiGraph object representing the argumentation framework,
        where nodes are strings of the argument indices (e.g., '1', '2').
    """

    graph = nx.DiGraph()

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Clean up the line
                line = line.strip()

                # Skip empty lines or comments
                if not line or line.startswith('#'):
                    continue

                #Handle the p-line
                if line.startswith('p af'):
                    try:
                        num_args = int(line.split()[2])
                        # Add all arguments as nodes. We use strings for node labels
                        # to avoid potential confusion with 0-based list indexing.
                        graph.add_nodes_from([str(i) for i in range(1, num_args + 1)])
                    except (IndexError, ValueError):
                        raise ValueError(f"Invalid p-line format: {line}")
                    
                # Handle attack lines
                else:
                    try:
                        attacker, attacked = line.split()
                        # add_edge will automatically add nodes if they don't exist,
                        # but parsing the p-line first is good practice.
                        graph.add_edge(attacker, attacked)
                    except ValueError:
                        print(f"Warning: Skipping malformed attack line: {line}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return graph # Return an empty graph

    return graph