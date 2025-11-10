"""Streamlit app for visualizing hypergraphs from JSON files."""

import json
import time
import streamlit as st
from pathlib import Path
import tempfile

from IsomorphismChecker_python_serial.diagram import Diagram, Orientation
from IsomorphismChecker_python_serial.graph_utils import create_hypergraph_from_data


def main():
    st.set_page_config(
        page_title="Hypergraph Visualizer", page_icon="🔗", layout="wide"
    )

    st.title("🔗 Hypergraph Isomorphism Checker")
    st.markdown("Upload a JSON file to visualize your hypergraph")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        orientation = st.selectbox(
            "Graph Orientation", options=["Top to Bottom", "Left to Right"], index=0
        )

        st.markdown("---")
        st.markdown("### 📝 JSON Format")
        st.code(
            """
{
  "nodes": [
    {"index": 0, "label": "A"},
    {"index": 1, "label": "B"}
  ],
  "edges": [
    {
      "sources": [0],
      "targets": [1],
      "label": "f",
      "index": 0
    }
  ],
  "input_nodes": [0],
  "output_nodes": [1]
}
        """,
            language="json",
        )

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a JSON file",
        type=["json"],
        help="Upload a JSON file containing hypergraph data",
    )

    # Example data button
    col1, col2 = st.columns([1, 4])
    with col1:
        use_example = st.button("📋 Use Example")

    if use_example or uploaded_file is not None:
        try:
            # Load JSON data
            if use_example:
                json_data = {
                    "nodes": [
                        {"index": 0, "label": "A"},
                        {"index": 1, "label": "B"},
                        {"index": 2, "label": "C"},
                    ],
                    "edges": [
                        {"sources": [0, 1], "targets": [2], "label": "f", "index": 0}
                    ],
                    "input_nodes": [0, 1],
                    "output_nodes": [2],
                }
            else:
                json_data = json.load(uploaded_file)  # type: ignore

            # Display JSON data
            with st.expander("📄 View JSON Data"):
                st.json(json_data)

            # Create hypergraph
            with st.spinner("Creating hypergraph..."):
                hg = create_hypergraph_from_data(json_data)

            # Validate hypergraph
            if not hg.is_valid():
                st.error("❌ Invalid hypergraph structure")
                time.sleep(2)
                st.rerun()

            st.success("✅ Hypergraph loaded successfully!")

            # Display graph info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nodes", len(hg.nodes))
            with col2:
                st.metric("Edges", len(hg.edges))
            with col3:
                st.metric("Input Nodes", len(hg.input_nodes))
            with col4:
                st.metric("Output Nodes", len(hg.output_nodes))

            # Create diagram
            st.subheader("📊 Hypergraph Visualization")

            orient = (
                Orientation.TOP_TO_BOTTOM
                if orientation == "Top to Bottom"
                else Orientation.LEFT_TO_RIGHT
            )

            with st.spinner("Rendering diagram..."):
                diagram = Diagram(openHyperGraph=hg, orientation=orient)

                # Create temporary file for rendering
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    diagram.render(str(tmp_path.with_suffix("")))

                    # Display the image
                    st.image(str(tmp_path), use_container_width=True)

                    # Cleanup
                    tmp_path.unlink(missing_ok=True)

            # Show diagram source
            with st.expander("🔍 View Graphviz Source"):
                st.code(diagram.source(), language="dot")

            # Download button
            st.download_button(
                label="💾 Download Diagram Source",
                data=diagram.source(),
                file_name="hypergraph.dot",
                mime="text/plain",
            )

            # Download PNG button
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                diagram.render(str(tmp_path.with_suffix("")))
                with open(tmp_path, "rb") as f:
                    png_data = f.read()

                st.download_button(
                    label="💾 Download Diagram PNG",
                    data=png_data,
                    file_name="hypergraph.png",
                    mime="image/png",
                )

                # Cleanup
                tmp_path.unlink(missing_ok=True)

        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {e}")
        except ValueError as e:
            st.error(f"❌ Error creating hypergraph: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
