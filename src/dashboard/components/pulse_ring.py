import math

def render_pulse_ring(risk_score: float, is_unknown: bool = False, size: int = 120) -> str:
    """
    Returns an SVG string for the animated Pulse Ring.
    
    Animation duration = max(0.3s, 3s * (1 - risk_score)).
    High risk = fast pulse. 
    Unknown = no animation, grey static rings.
    """
    if is_unknown:
        color = "#4B5563" # --unknown-grey
        duration = 0 # Static
    else:
        if risk_score >= 0.7:
            color = "#E5383B" # --risk-high
        elif risk_score >= 0.4:
            color = "#F4A261" # --risk-mid
        else:
            color = "#52B788" # --risk-low
            
        duration = max(0.3, 3.0 * (1.0 - risk_score))
        
    center = size // 2
    r_base = int(size * 0.25)
    
    if is_unknown or duration == 0:
        # Static rings
        return f"""
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
            <circle cx="{center}" cy="{center}" r="{r_base}" fill="none" stroke="{color}" stroke-width="2" opacity="1" />
            <circle cx="{center}" cy="{center}" r="{r_base + 10}" fill="none" stroke="{color}" stroke-width="1" opacity="0.5" />
            <circle cx="{center}" cy="{center}" r="{r_base + 20}" fill="none" stroke="{color}" stroke-width="0.5" opacity="0.2" />
        </svg>
        """
        
    # Animated rings (using SMIL animations within SVG)
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
        <circle cx="{center}" cy="{center}" r="{r_base}" fill="none" stroke="{color}" stroke-width="3">
            <animate attributeName="r" values="{r_base};{r_base+5};{r_base}" dur="{duration}s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="1;0.8;1" dur="{duration}s" repeatCount="indefinite" />
        </circle>
        
        <circle cx="{center}" cy="{center}" r="{r_base+10}" fill="none" stroke="{color}" stroke-width="1.5">
            <animate attributeName="r" values="{r_base+10};{r_base+20};{r_base+10}" dur="{duration}s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.6;0.2;0.6" dur="{duration}s" repeatCount="indefinite" />
        </circle>
        
        <circle cx="{center}" cy="{center}" r="{r_base+20}" fill="none" stroke="{color}" stroke-width="0.5">
            <animate attributeName="r" values="{r_base+20};{r_base+35};{r_base+20}" dur="{duration}s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.3;0.0;0.3" dur="{duration}s" repeatCount="indefinite" />
        </circle>
    </svg>
    """
