def get_risk_color(score: float, is_unknown: bool = False) -> str:
    """Map risk score to the exact token hex color."""
    if is_unknown:
        return "var(--unknown-grey)"
    if score >= 0.7:
        return "var(--risk-high)"
    elif score >= 0.4:
        return "var(--risk-mid)"
    else:
        return "var(--risk-low)"

def get_risk_label(score: float, is_unknown: bool = False) -> str:
    """Map risk score to string label."""
    if is_unknown:
        return "unknown"
    if score >= 0.7:
        return "illicit"
    elif score >= 0.4:
        return "ambiguous"
    else:
        return "licit"

def render_risk_pill(score: float, is_unknown: bool = False) -> str:
    """
    Returns an HTML string for the Risk Pill component.
    Format specified in frontend-design.md: 
    Font: Inter 11px medium. Padding: 2px 8px. Border radius: 4px.
    """
    color = get_risk_color(score, is_unknown)
    label = get_risk_label(score, is_unknown).upper()
    
    if is_unknown:
        bg_color = "transparent"
        text_color = "var(--text-muted)"
    else:
        # Use CSS color-mix or rgba approximation. Streamlit passes hex, so we'll use inline styles 
        # with opacity to approximate the 20% bg. A simple hack is just a dark semi-transparent bg.
        bg_color = f"color-mix(in srgb, {color} 20%, transparent)"
        text_color = color
        
    html = f"""
    <div style="
        display: inline-block;
        background-color: {bg_color};
        border: 1px solid {color};
        color: {text_color};
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    ">
        {label}
    </div>
    """
    return html
