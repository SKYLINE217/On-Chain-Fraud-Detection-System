import streamlit as st
from pathlib import Path

# Must be the first Streamlit command
st.set_page_config(
    page_title="Forensic Dashboard | On-Chain Fraud",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def inject_custom_css():
    """Inject the custom CSS tokens and font imports from style.css."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    inject_custom_css()
    
    # ── Global Header ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("<h3 style='margin:0; padding-top:8px;'>On-Chain Forensics</h3>", unsafe_allow_html=True)
    with col2:
        search_query = st.text_input("Search Wallet Address / TxID", placeholder="e.g. 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", label_visibility="collapsed")
    with col3:
        st.markdown("<div style='text-align:right; color:var(--text-muted); padding-top:12px; font-family:\"IBM Plex Mono\"; font-size:11px;'>Scores updated 2h ago</div>", unsafe_allow_html=True)
        
    st.markdown("<div style='border-bottom: 1px solid var(--border-subtle); margin-bottom: 24px;'></div>", unsafe_allow_html=True)
    
    # Handle global search
    if search_query:
        st.session_state.selected_wallet = search_query

    # ── Navigation & Tabs ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Cluster Explorer", "Wallet Lookup", "Transaction Path"])
    
    with tab1:
        from pages.overview import render_overview
        render_overview()
        
    with tab2:
        from pages.clusters import render_clusters
        render_clusters()
        
    with tab3:
        from pages.wallet import render_wallet
        # If a search occurred, default to looking up that wallet
        wallet_to_lookup = st.session_state.get("selected_wallet", None)
        render_wallet(wallet_to_lookup)
        
    with tab4:
        from pages.paths import render_paths
        render_paths()

if __name__ == "__main__":
    main()
