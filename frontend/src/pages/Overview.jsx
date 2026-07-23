import React, { useEffect, useState } from 'react';
import { Card } from '../components/Shared';
import { Activity, Database, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function Overview() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="animate-fade-in">
      <h1 className="page-title" style={{ marginBottom: '1.5rem' }}>System Overview</h1>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        <Card hover>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
            <div style={{ padding: '0.75rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '12px', color: '#3b82f6' }}>
              <Activity size={24} />
            </div>
            <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-secondary)' }}>API Status</h3>
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 600 }}>
            {health?.status === 'healthy' ? <span style={{color: 'var(--success)'}}>Online</span> : <span style={{color: 'var(--danger)'}}>Offline</span>}
          </div>
        </Card>

        <Card hover className="delay-100 animate-fade-in">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
            <div style={{ padding: '0.75rem', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '12px', color: '#8b5cf6' }}>
              <Database size={24} />
            </div>
            <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-secondary)' }}>Graph Nodes</h3>
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 600 }}>203,769</div>
        </Card>

        <Card hover className="delay-200 animate-fade-in">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
            <div style={{ padding: '0.75rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '12px', color: '#ef4444' }}>
              <AlertTriangle size={24} />
            </div>
            <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-secondary)' }}>High Risk</h3>
          </div>
          <div style={{ fontSize: '1.75rem', fontWeight: 600 }}>~2.1%</div>
        </Card>
      </div>

      <Card>
        <h2 style={{ marginBottom: '1rem', fontSize: '1.25rem' }}>Welcome to On-Chain Forensics</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
          This dashboard provides real-time risk scoring and explainability for blockchain transactions. 
          It utilizes a GraphSAGE neural network trained on the Elliptic dataset to detect illicit activity 
          based on both transaction features and graph topology.
        </p>
        <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--success)' }}>
            <ShieldCheck size={20} />
            <span>GNN Scoring Active</span>
          </div>
        </div>
      </Card>
    </div>
  );
}
