"""Main module for the data-parallel proof checker."""

import logging

from pathlib import Path
from IsomorphismChecker_python_serial.isomorphisms import MC_isomorphism
from IsomorphismChecker_python_serial.graph_utils import create_hypergraph
from IsomorphismChecker_python_serial.draw import draw_graph

import argparse

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def command_line_interface():
    parser = argparse.ArgumentParser(
        description="Data-Parallel Proof Checker Command Line Interface"
    )
    parser.add_argument(
        "--graph1",
        type=str,
        required=True,
        help="Path to the first hypergraph JSON file.",
    )
    parser.add_argument(
        "--graph2",
        type=str,
        required=True,
        help="Path to the second hypergraph JSON file.",
    )

    parser.add_argument(
        "--output1",
        type=str,
        default="",
        help="Path to the output diagram file for graph1.",
    )
    parser.add_argument(
        "--output2",
        type=str,
        default="",
        help="Path to the output diagram file for graph2.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args()
    return args


def main():

    args = command_line_interface()

    if not args.graph1 or not args.graph2:
        print("Error: Both graph1 and graph2 arguments are required.")
        return

    if not Path(args.graph1).is_file():
        print(f"Error: The file {args.graph1} does not exist.")
        return

    if not Path(args.graph2).is_file():
        print(f"Error: The file {args.graph2} does not exist.")
        return

    g1 = create_hypergraph(Path(args.graph1))
    g2 = create_hypergraph(Path(args.graph2))

    isomorphic, p_nodes, p_edges = MC_isomorphism(g1, g2)

    if isomorphic:
        print("The two hypergraphs are isomorphic.")
        logger.info("Node mapping: %s", p_nodes)
        logger.info("Edge mapping: %s", p_edges)
    else:
        logger.info("The two hypergraphs are not isomorphic.")

    if args.output1:
        logger.info("Rendering the first hypergraph to %s", args.output1)
        draw_graph(g1, args.output1, args.verbose)
        logger.info("Rendering completed.")

    if args.output2:
        logger.info("Rendering the second hypergraph to %s", args.output2)
        draw_graph(g2, args.output2, args.verbose)
        logger.info("Rendering completed.")


if __name__ == "__main__":
    main()
