import React from 'react';

export function Card({ children, className = '', hover = false, ...props }) {
  const baseClass = hover ? 'glass-card' : 'glass-panel';
  return (
    <div className={`${baseClass} ${className}`} {...props} style={{ padding: '1.5rem', ...props.style }}>
      {children}
    </div>
  );
}

export function RiskBadge({ score, label, className = '' }) {
  let badgeClass = 'badge-unknown';
  let displayLabel = label || 'Unknown';
  
  if (score !== undefined && score !== null) {
    if (score >= 0.7) {
      badgeClass = 'badge-high-risk';
      displayLabel = 'High Risk';
    } else if (score < 0.7) {
      badgeClass = 'badge-low-risk';
      displayLabel = 'Low Risk';
    }
  }

  // Override if label explicitly passed
  if (label && label.toLowerCase() === 'illicit') badgeClass = 'badge-high-risk';
  if (label && label.toLowerCase() === 'licit') badgeClass = 'badge-low-risk';
  if (label && label.toLowerCase() === 'unknown') badgeClass = 'badge-unknown';

  return (
    <span className={`badge ${badgeClass} ${className}`}>
      {score !== undefined && score !== null ? `${displayLabel} (${(score * 100).toFixed(1)}%)` : displayLabel}
    </span>
  );
}

export function LoadingSpinner() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '2rem' }}>
      <svg className="animate-spin" style={{ width: '2rem', height: '2rem', color: 'url(#gradient-spinner)', animation: 'spin 1s linear infinite' }} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <defs>
          <linearGradient id="gradient-spinner" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0075FF" />
            <stop offset="100%" stopColor="#17C1E8" />
          </linearGradient>
        </defs>
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.1)" strokeWidth="4"></circle>
        <path className="opacity-75" fill="url(#gradient-spinner)" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
