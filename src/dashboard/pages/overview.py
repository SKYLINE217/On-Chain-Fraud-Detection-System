import streamlit as st
import plotly.graph_objects as go
import numpy as np

def render_overview():
    st.markdown("### System Health Overview")
    
    # ── Stat Tiles ───────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div data-testid="stMetricLabel">Flagged Wallets</div>
            <div data-testid="stMetricValue" style="color: var(--risk-high);">1,248</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div data-testid="stMetricLabel">Avg Confidence</div>
            <div data-testid="stMetricValue">91.4%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div data-testid="stMetricLabel">High-Risk Clusters</div>
            <div data-testid="stMetricValue">32</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-container">
            <div data-testid="stMetricLabel">Last Batch Time</div>
            <div data-testid="stMetricValue">14m ago</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1])
    
    # ── Risk Distribution Histogram ─────────────────────────────────
    with col_left:
        st.markdown("#### Risk Score Distribution")
        # Mock data for histogram
        x = np.random.beta(0.5, 0.5, 1000)
        
        fig = go.Figure()
        # Low risk
        low = x[x < 0.4]
        fig.add_trace(go.Histogram(x=low, name='Licit', marker_color='#52B788', opacity=0.8, xbins=dict(start=0.0, end=1.0, size=0.05)))
        # Mid risk
        mid = x[(x >= 0.4) & (x < 0.7)]
        fig.add_trace(go.Histogram(x=mid, name='Ambiguous', marker_color='#F4A261', opacity=0.8, xbins=dict(start=0.0, end=1.0, size=0.05)))
        # High risk
        high = x[x >= 0.7]
        fig.add_trace(go.Histogram(x=high, name='Illicit', marker_color='#E5383B', opacity=0.8, xbins=dict(start=0.0, end=1.0, size=0.05)))
        
        fig.update_layout(
            barmode='stack',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            height=250,
            xaxis=dict(showgrid=False, zeroline=False, color='#6B7A90'),
            yaxis=dict(showgrid=True, gridcolor='#1E2A38', zeroline=False, color='#6B7A90')
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── PR-AUC Baseline Comparison ──────────────────────────────────
    with col_right:
        st.markdown("#### PR-AUC vs Baselines (Test Set)")
        
        models = ['LogReg', 'Random Forest', 'XGBoost', 'GraphSAGE', 'GAT']
        scores = [0.42, 0.61, 0.78, 0.82, 0.80]
        colors = ['#1E2A38', '#1E2A38', '#1E2A38', '#38BDF8', '#38BDF8']
        
        fig2 = go.Figure(go.Bar(
            x=scores,
            y=models,
            orientation='h',
            marker_color=colors,
            text=[f"{s:.2f}" for s in scores],
            textposition='auto',
            textfont=dict(family="IBM Plex Mono", color="#E8EDF2")
        ))
        
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            height=250,
            xaxis=dict(showgrid=True, gridcolor='#1E2A38', color='#6B7A90', range=[0, 1]),
            yaxis=dict(showgrid=False, color='#E8EDF2', categoryorder='total ascending')
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    # ── Temporal F1 Line Chart ──────────────────────────────────────
    st.markdown("#### F1 Score Temporal Degradation (Steps 35-49)")
    
    steps = list(range(35, 50))
    xgb_f1 = np.linspace(0.85, 0.65, 15) + np.random.normal(0, 0.02, 15)
    gsage_f1 = np.linspace(0.88, 0.75, 15) + np.random.normal(0, 0.02, 15)
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=steps, y=xgb_f1, mode='lines+markers', name='XGBoost', line=dict(color='#F4A261', dash='dash')))
    fig3.add_trace(go.Scatter(x=steps, y=gsage_f1, mode='lines+markers', name='GraphSAGE (Combined)', line=dict(color='#38BDF8')))
    
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#E8EDF2")),
        xaxis=dict(title="Time Step", showgrid=False, color='#6B7A90'),
        yaxis=dict(title="F1 Score", showgrid=True, gridcolor='#1E2A38', color='#6B7A90', range=[0.4, 1.0])
    )
    st.plotly_chart(fig3, use_container_width=True)
