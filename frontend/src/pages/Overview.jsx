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
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        <Card hover>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '0.875rem', color: 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase' }}>API Status</h3>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.5rem' }}>
                {health?.status === 'healthy' ? <span style={{color: 'var(--success)'}}>Online</span> : <span style={{color: 'var(--danger)'}}>Offline</span>}
              </div>
            </div>
            <div style={{ padding: '0.85rem', background: 'var(--accent-gradient)', borderRadius: '12px', color: '#fff', boxShadow: '0 4px 10px rgba(0, 117, 255, 0.4)' }}>
              <Activity size={24} />
            </div>
          </div>
          <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
            <span style={{ color: 'var(--success)' }}>+2%</span> since last hour
          </div>
        </Card>

        <Card hover className="delay-100 animate-fade-in">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '0.875rem', color: 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase' }}>Graph Nodes</h3>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.5rem' }}>
                203,769
              </div>
            </div>
            <div style={{ padding: '0.85rem', background: 'var(--accent-purple)', borderRadius: '12px', color: '#fff', boxShadow: '0 4px 10px rgba(134, 140, 255, 0.4)' }}>
              <Database size={24} />
            </div>
          </div>
          <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
            <span style={{ color: 'var(--success)' }}>+1.5%</span> from last week
          </div>
        </Card>

        <Card hover className="delay-200 animate-fade-in">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '0.875rem', color: 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase' }}>High Risk</h3>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '0.5rem' }}>
                ~2.1%
              </div>
            </div>
            <div style={{ padding: '0.85rem', background: 'var(--danger-gradient)', borderRadius: '12px', color: '#fff', boxShadow: '0 4px 10px rgba(227, 26, 26, 0.4)' }}>
              <AlertTriangle size={24} />
            </div>
          </div>
          <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
            <span style={{ color: 'var(--danger)' }}>+0.3%</span> recent surge
          </div>
        </Card>
      </div>

      <Card style={{ 
        position: 'relative', 
        overflow: 'hidden',
        padding: '3rem',
      }}>
        <div style={{ 
          position: 'absolute', 
          top: 0, right: 0, bottom: 0, left: '50%',
          background: 'radial-gradient(circle at 100% 50%, rgba(0, 117, 255, 0.15), transparent 70%)',
          pointerEvents: 'none'
        }} />
        <h2 style={{ marginBottom: '1rem', fontSize: '1.5rem' }}>Welcome to On-Chain Forensics</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', maxWidth: '600px', lineHeight: 1.6 }}>
          This dashboard provides real-time risk scoring and explainability for blockchain transactions. 
          It utilizes a GraphSAGE neural network trained on the Elliptic dataset to detect illicit activity 
          based on both transaction features and graph topology.
        </p>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-primary)', background: 'rgba(1, 181, 116, 0.2)', padding: '0.5rem 1rem', borderRadius: '999px', border: '1px solid rgba(1, 181, 116, 0.4)' }}>
            <ShieldCheck size={20} color="var(--success)" />
            <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>GNN Scoring Active</span>
          </div>
        </div>
      </Card>
    </div>
  );
}
