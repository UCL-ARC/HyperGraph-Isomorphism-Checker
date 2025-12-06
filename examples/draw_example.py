from pathlib import Path
from IsomorphismChecker_python_serial.graph_utils import create_hypergraph
from IsomorphismChecker_python_serial.draw import draw_graph

import argparse

parser = argparse.ArgumentParser(description="Draw a hypergraph from a JSON file.")

parser.add_argument(
    "graph_file", type=str, help="Path to the JSON file containing the hypergraph."
)
parser.add_argument(
    "--highlighted_node",
    type=str,
    help="List of node indices to highlight in the diagram (Comma-separated).",
)

parser.add_argument(
    "--highlighted_edges",
    type=str,
    help="List of edge indices to highlight in the diagram (Comma-separated).",
)

args = parser.parse_args()

hypergraph = create_hypergraph(Path(args.graph_file))

draw_graph(
    hypergraph,
    highlighted_nodes=(
        [int(n) for n in args.highlighted_node.split(",")]
        if args.highlighted_node
        else []
    ),
    highlighted_edges=(
        [int(e) for e in args.highlighted_edges.split(",")]
        if args.highlighted_edges
        else []
    ),
)
