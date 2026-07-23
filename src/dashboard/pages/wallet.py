import streamlit as st
import plotly.graph_objects as go
from pyvis.network import Network
import pandas as pd
import numpy as np

from components.pulse_ring import render_pulse_ring
from components.risk_pill import render_risk_pill
from components.graphs import render_pyvis_graph
from utils import load_mock_data, get_mock_risk_score, get_mock_shap_values

def render_wallet(wallet_id: str = None):
    # If no wallet selected, prompt the user
    if not wallet_id:
        st.markdown("<div style='text-align:center; padding: 40px; color: var(--text-muted);'>Search for a wallet address or TxID to view forensics.</div>", unsafe_allow_html=True)
        return
        
    df = load_mock_data()
    risk_score = get_mock_risk_score(wallet_id, df)
    is_unknown = (risk_score == 0.5) # Using 0.5 as proxy for unknown for this mock
    
    # ── Header ───────────────────────────────────────────────────────
    col_ring, col_meta = st.columns([1, 4])
    
    with col_ring:
        st.markdown(render_pulse_ring(risk_score, is_unknown=is_unknown, size=120), unsafe_allow_html=True)
        
    with col_meta:
        st.markdown(f"<div class='mono-text' style='font-size: 48px; font-weight: 600; line-height: 1.1;'>RISK: {risk_score:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex; gap:12px; align-items:center; margin-bottom: 8px;'>{render_risk_pill(risk_score, is_unknown)} <span class='mono-text' style='color:var(--text-muted); font-size:14px;'>CONFIDENCE: {np.random.uniform(70,99):.1f}%</span></div>", unsafe_allow_html=True)
        
        # Confidence Bar
        bar_color = "var(--unknown-grey)" if is_unknown else ("var(--risk-high)" if risk_score >= 0.7 else ("var(--risk-mid)" if risk_score >= 0.4 else "var(--risk-low)"))
        st.markdown(f"""
        <div style="width:100%; background-color:var(--bg-raised); height:4px; border-radius:2px; margin-bottom: 8px;">
            <div style="width:{risk_score*100}%; background-color:{bar_color}; height:100%; border-radius:2px;"></div>
        </div>
        <div style='color:var(--text-muted); font-size:13px;'>Community #441 &middot; Time step 37</div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: var(--border-subtle); margin: 32px 0;'>", unsafe_allow_html=True)
    
    # ── Details Pane (SHAP & Graph) ──────────────────────────────────
    col_shap, col_graph = st.columns([1, 1])
    
    with col_shap:
        st.markdown("#### Top SHAP Features")
        features, values = get_mock_shap_values()
        
        colors = ['#E5383B' if v > 0 else '#52B788' for v in values]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=350,
            xaxis=dict(showgrid=True, gridcolor='#1E2A38', color='#6B7A90', title="SHAP Value"),
            yaxis=dict(showgrid=False, color='#E8EDF2', tickfont=dict(family="IBM Plex Mono", size=11))
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Rationale")
        st.markdown("""
        <div style="background-color:var(--bg-panel); padding: 16px; border-radius: 4px; border: 1px solid var(--border-subtle); font-size: 14px;">
            Flagged due to: <span class='mono-text'>burst_score</span> 3.2σ above baseline; connected to 2 known illicit clusters; <span class='mono-text'>address_age</span> < 2 time steps.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Request Explanation", type="secondary"):
            st.info("Running GNNExplainer… (5–15s expected)")

    with col_graph:
        st.markdown("#### 2-Hop Ego-Graph")
        st.caption("Showing 12 of 12 nodes (max 200)")
        
        net = Network(height="450px", width="100%", bgcolor="#0A0E14", font_color="#E8EDF2")
        
        # Add central node
        net.add_node(wallet_id, label=wallet_id[:6]+"...", color="#E5383B", size=20, borderWidth=3, borderColor="#38BDF8")
        
        # Add mock neighbors
        for i in range(11):
            n_id = f"neighbor_{i}"
            r = np.random.uniform(0, 1)
            c = "#E5383B" if r > 0.7 else ("#F4A261" if r > 0.4 else "#52B788")
            net.add_node(n_id, label=f"1x{i}...", color=c, size=10)
            
            # Edges
            if np.random.rand() > 0.5:
                net.add_edge(wallet_id, n_id, color="#1E2A38")
            else:
                net.add_edge(n_id, wallet_id, color="#1E2A38")
                
        render_pyvis_graph(net, height=450)
