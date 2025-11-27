import argparse


from IsomorphismChecker_python_serial.random_graph_generator import (
    generate_random_hypergraph,
    graph_to_json_serializable,
)

parser = argparse.ArgumentParser(
    description="Generate a random hypergraph and save it as a JSON file."
)

parser.add_argument(
    "--num_nodes",
    type=int,
    default=100,
    help="Number of nodes (wires) in the hypergraph.",
)
parser.add_argument(
    "--num_edges",
    type=int,
    default=20,
    help="Number of hyperedges (boxes) in the hypergraph.",
)
parser.add_argument(
    "--num_inputs",
    type=int,
    default=30,
    help="Number of input wires in the hypergraph.",
)
parser.add_argument(
    "--num_outputs",
    type=int,
    default=40,
    help="Number of output wires in the hypergraph.",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility."
)
parser.add_argument(
    "--output_file",
    type=str,
    default="random_hypergraph_auto",
    help="Output JSON file name.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="trial_graphs",
    help="Directory to save the output JSON file.",
)

args = parser.parse_args()

num_nodes = args.num_nodes
num_edges = args.num_edges
num_inputs = args.num_inputs
num_outputs = args.num_outputs
output_file = args.output_file
output_dir = args.output_dir
seed = args.seed

hypergraph = generate_random_hypergraph(
    num_nodes=num_nodes,
    num_edges=num_edges,
    num_inputs=num_inputs,
    num_outputs=num_outputs,
    seed=seed,
)
graph_to_json_serializable(hypergraph, file_name=output_file, directory=output_dir)
