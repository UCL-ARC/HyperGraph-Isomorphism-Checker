#!/bin/zsh

uv run python examples/draw_example.py --highlighted_edge 30 tests/inputs/DrugALarge.json 
mv hypergraph_diagram.png GraphA.png

uv run python examples/draw_example.py --highlighted_edge 30 tests/inputs/DrugBLarge.json 
mv hypergraph_diagram.png GraphB.png