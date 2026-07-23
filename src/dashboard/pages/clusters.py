import streamlit as st
import pandas as pd
import numpy as np
from pyvis.network import Network
from components.graphs import render_pyvis_graph
from components.risk_pill import render_risk_pill

def render_clusters():
    st.markdown("### Discovered Illicit Communities")
    
    # ── Filter Row ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        min_size = st.selectbox("Min cluster size", [5, 10, 50, 100], index=1)
    with col2:
        min_risk = st.selectbox("Min avg risk", [0.4, 0.5, 0.7, 0.9], index=2)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_table, col_graph = st.columns([3, 2])
    
    with col_table:
        st.markdown("#### Cluster Directory")
        # Mock cluster data
        data = []
        for i in range(15):
            size = np.random.randint(10, 200)
            risk = np.random.uniform(0.3, 0.95)
            flagged = int(size * np.random.uniform(0.1, risk))
            data.append({
                "Community ID": f"#{400 + i}",
                "Size": size,
                "Avg Risk Score": risk,
                "Flagged Nodes": flagged,
                "Primary Label": "illicit" if risk > 0.7 else "ambiguous"
            })
            
        df = pd.DataFrame(data).sort_values("Avg Risk Score", ascending=False).reset_index(drop=True)
        
        # Display as a dense HTML table to apply tokens properly
        html = "<table style='width:100%; border-collapse: collapse; font-size:14px;'>"
        html += """
        <thead>
            <tr style='border-bottom: 1px solid var(--border-subtle); color: var(--text-muted); font-family: "IBM Plex Mono", monospace; font-size: 11px; text-transform: uppercase;'>
                <th style='padding: 8px; text-align: left;'>Community ID</th>
                <th style='padding: 8px; text-align: right;'>Size</th>
                <th style='padding: 8px; text-align: center;'>Avg Risk</th>
                <th style='padding: 8px; text-align: right;'>Flagged</th>
            </tr>
        </thead>
        <tbody>
        """
        
        for i, row in df.iterrows():
            bg = "var(--bg-raised)" if i % 2 == 0 else "var(--bg-panel)"
            html += f"""
            <tr style='background-color: {bg}; border-bottom: 1px solid var(--border-subtle);'>
                <td style='padding: 10px 8px; font-family: "IBM Plex Mono", monospace; color: var(--accent-flow);'>{row['Community ID']}</td>
                <td style='padding: 10px 8px; text-align: right;'>{row['Size']}</td>
                <td style='padding: 10px 8px; text-align: center;'>{render_risk_pill(row['Avg Risk Score'])}</td>
                <td style='padding: 10px 8px; text-align: right; color: var(--risk-high); font-weight: 500;'>{row['Flagged Nodes']}</td>
            </tr>
            """
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)
        
    with col_graph:
        st.markdown("#### Ego-Subgraph (Selected)")
        st.caption("Community #400 (cose-bilkent layout)")
        
        net = Network(height="500px", width="100%", bgcolor="#0A0E14", font_color="#E8EDF2")
        
        # Build mock cluster graph
        nodes = [f"n{i}" for i in range(25)]
        for n in nodes:
            r = np.random.uniform(0.5, 0.99)
            c = "#E5383B" if r > 0.7 else "#F4A261"
            net.add_node(n, size=np.random.randint(5, 20), color=c)
            
        for _ in range(40):
            u = np.random.choice(nodes)
            v = np.random.choice(nodes)
            if u != v:
                net.add_edge(u, v, color="#1E2A38")
                
        render_pyvis_graph(net, height=500)
