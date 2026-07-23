import streamlit as st
import numpy as np
from pyvis.network import Network
from components.graphs import render_pyvis_graph
from components.risk_pill import render_risk_pill

def render_paths():
    st.markdown("### Transaction Path Analysis")
    st.caption("Discover layering and laundering chains between known entities.")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.text_input("Source Wallet", value="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
    with col2:
        st.text_input("Destination Wallet", value="3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy")
    with col3:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        st.button("Find Path", type="primary", use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Path Graph (Shortest: 4 Hops)")
    
    # Render Dagre (hierarchical left-to-right) mock graph
    net = Network(height="400px", width="100%", bgcolor="#0A0E14", font_color="#E8EDF2", directed=True)
    
    # Custom options for Dagre layout
    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "nodeSpacing": 150,
          "levelSeparation": 200
        }
      },
      "nodes": {
        "borderWidth": 2,
        "font": { "color": "#E8EDF2", "face": "IBM Plex Mono", "size": 14 }
      },
      "edges": {
        "color": { "color": "#38BDF8" },
        "font": { "color": "#6B7A90", "face": "IBM Plex Mono", "size": 11, "align": "top" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
        "smooth": { "type": "cubicBezier", "forceDirection": "horizontal" }
      }
    }
    """)
    
    # Add path nodes
    path = [
        {"id": "src", "label": "1A1zP1...", "risk": 0.8},
        {"id": "h1", "label": "1BvBMS...", "risk": 0.92},
        {"id": "h2", "label": "3K28t...", "risk": 0.45},
        {"id": "dst", "label": "3J98t1...", "risk": 0.12},
    ]
    
    for i, n in enumerate(path):
        c = "#E5383B" if n["risk"] > 0.7 else ("#F4A261" if n["risk"] > 0.4 else "#52B788")
        net.add_node(n["id"], label=n["label"], color=c, level=i)
        
    # Add path edges with amounts/times
    net.add_edge("src", "h1", label="12.5 BTC\nStep 34")
    net.add_edge("h1", "h2", label="12.4 BTC\nStep 35")
    net.add_edge("h2", "dst", label="12.0 BTC\nStep 36")
    
    import tempfile
    import streamlit.components.v1 as components
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        path_file = tmp.name
    net.save_graph(path_file)
    with open(path_file, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=400)
