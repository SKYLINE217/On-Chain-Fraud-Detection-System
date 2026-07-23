import tempfile
from pathlib import Path
from pyvis.network import Network
import streamlit.components.v1 as components

def get_pyvis_options():
    """Returns a JSON string of PyVis options for the dark forensic theme."""
    return """
    var options = {
      "nodes": {
        "borderWidth": 1,
        "borderWidthSelected": 3,
        "font": {
          "color": "#E8EDF2",
          "face": "IBM Plex Mono",
          "size": 12
        },
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.5)",
          "size": 10,
          "x": 0,
          "y": 0
        }
      },
      "edges": {
        "color": {
          "color": "#1E2A38",
          "highlight": "#38BDF8",
          "hover": "#38BDF8"
        },
        "smooth": {
          "type": "continuous",
          "forceDirection": "none"
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 50
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200
      }
    }
    """

def render_pyvis_graph(net: Network, height: int = 500):
    """
    Renders a PyVis network in Streamlit.
    Works around Streamlit's iframe limitations by saving to a temp file and injecting.
    """
    net.set_options(get_pyvis_options())
    
    # PyVis needs an explicit HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        path = tmp.name
        
    net.save_graph(path)
    
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
        
    components.html(html, height=height)
