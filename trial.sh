#!/bin/zsh

uv run python examples/draw_example.py --highlighted_edge 71,56,28 tests/inputs/DrugALarge.json 
mv hypergraph_diagram.png GraphA.png

uv run python examples/draw_example.py --highlighted_edge 71,56,28 tests/inputs/DrugBLarge.json 
mv hypergraph_diagram.png GraphB.png