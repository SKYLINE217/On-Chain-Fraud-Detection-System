import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Card, RiskBadge, LoadingSpinner } from '../components/Shared';
import GraphVisualizer from '../components/GraphVisualizer';
import { apiClient } from '../api/client';
import { AlertCircle, FileText, Network } from 'lucide-react';

export default function WalletLookup() {
  const [searchParams] = useSearchParams();
  const address = searchParams.get('address');
  
  const [wallet, setWallet] = useState(null);
  const [subgraph, setSubgraph] = useState(null);
  const [explanation, setExplanation] = useState(null);
  
  const [loading, setLoading] = useState(false);
  const [explainLoading, setExplainLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!address) return;
    
    setLoading(true);
    setError(null);
    setWallet(null);
    setSubgraph(null);
    setExplanation(null);

    Promise.all([
      apiClient.getWallet(address),
      apiClient.getSubgraph(address)
    ])
      .then(([walletData, subgraphData]) => {
        setWallet(walletData);
        setSubgraph(subgraphData);
      })
      .catch(err => {
        setError(err.message);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [address]);

  const handleExplain = () => {
    if (!address) return;
    setExplainLoading(true);
    apiClient.explainPrediction(address)
      .then(data => setExplanation(data))
      .catch(err => console.error(err))
      .finally(() => setExplainLoading(false));
  };

  if (!address) {
    return (
      <div className="animate-fade-in" style={{ textAlign: 'center', padding: '4rem 2rem', color: 'var(--text-muted)' }}>
        <Search size={48} style={{ opacity: 0.5, marginBottom: '1rem' }} />
        <h2>Search for a Wallet</h2>
        <p>Use the search bar above to look up a transaction or wallet address.</p>
      </div>
    );
  }

  if (loading) return <LoadingSpinner />;
  
  if (error) {
    return (
      <Card className="animate-fade-in" style={{ borderColor: 'var(--danger)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', color: 'var(--danger)' }}>
          <AlertCircle />
          <h3>Error Loading Wallet</h3>
        </div>
        <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>{error}</p>
      </Card>
    );
  }

  if (!wallet) return null;

  return (
    <div className="animate-fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
        <div>
          <h1 className="page-title" style={{ fontFamily: 'monospace', fontSize: '1.5rem', marginBottom: '0.5rem' }}>
            {wallet.txId}
          </h1>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <RiskBadge score={wallet.risk_score} label={wallet.predicted_label || wallet.txClass} />
            {wallet.communityId && <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>Cluster #{wallet.communityId}</span>}
          </div>
        </div>
        {!explanation && (
          <button 
            onClick={handleExplain}
            disabled={explainLoading}
            style={{ 
              background: 'var(--accent-primary)', 
              color: 'white', 
              border: 'none', 
              padding: '0.5rem 1.25rem', 
              borderRadius: '8px', 
              cursor: 'pointer',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            {explainLoading ? <LoadingSpinner /> : <><FileText size={18} /> Generate Explanation</>}
          </button>
        )}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '1.5rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <Card>
            <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Network size={20} />
              Local Subgraph
            </h3>
            {subgraph ? (
              <GraphVisualizer nodes={subgraph.nodes} edges={subgraph.edges} centerNodeId={wallet.txId} height={400} />
            ) : <LoadingSpinner />}
          </Card>
          
          {explanation && (
            <Card className="animate-fade-in delay-100">
              <h3 style={{ marginBottom: '1rem' }}>AI Rationale</h3>
              <p style={{ color: 'var(--text-primary)', lineHeight: 1.6, background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px' }}>
                {explanation.rationale}
              </p>
            </Card>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <Card>
            <h3 style={{ marginBottom: '1rem' }}>Metadata</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.875rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Time Step</span>
                <span>{wallet.timeStep}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>PageRank</span>
                <span>{wallet.pageRank?.toFixed(5) || 'N/A'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Risk Score</span>
                <span style={{ fontWeight: 600 }}>{wallet.risk_score?.toFixed(4) || 'N/A'}</span>
              </div>
            </div>
          </Card>

          {explanation && explanation.shap_top_features && (
            <Card className="animate-fade-in delay-200">
              <h3 style={{ marginBottom: '1rem' }}>Key Features</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {explanation.shap_top_features.map((f, i) => (
                  <div key={i} style={{ background: 'rgba(255,255,255,0.03)', padding: '0.75rem', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                      <span style={{ fontWeight: 500, fontSize: '0.875rem' }}>{f.feature_name}</span>
                      <span style={{ fontSize: '0.875rem', color: f.shap_value > 0 ? 'var(--danger)' : 'var(--success)' }}>
                        {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(3)}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Value: {f.feature_value.toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
// Import Search icon inside WalletLookup just for empty state
import { Search } from 'lucide-react';
