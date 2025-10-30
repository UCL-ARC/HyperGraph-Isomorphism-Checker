from pathlib import Path
from IsomorphismChecker_python_serial.graph_utils import create_hypergraph
from IsomorphismChecker_python_serial.draw import draw_graph

import argparse

parser = argparse.ArgumentParser(description="Draw a hypergraph from a JSON file.")
parser.add_argument(
    "graph_file", type=str, help="Path to the JSON file containing the hypergraph."
)
args = parser.parse_args()

hypergraph = create_hypergraph(Path(args.graph_file))

draw_graph(hypergraph)
