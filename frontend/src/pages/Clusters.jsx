import React, { useEffect, useState } from 'react';
import { Card, LoadingSpinner } from '../components/Shared';
import { apiClient } from '../api/client';
import { useNavigate } from 'react-router-dom';
import { Share2, Users } from 'lucide-react';

export default function Clusters() {
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    apiClient.getTopClusters(20)
      .then(data => setClusters(data))
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner />;

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">Top High-Risk Clusters</h1>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.5rem' }}>
        {clusters.map((cluster, i) => (
          <Card 
            key={cluster.cluster_id} 
            hover 
            className={`delay-${(i % 5) * 100} animate-fade-in`}
            style={{ cursor: 'pointer', padding: '1.5rem' }}
            onClick={() => alert('Cluster detail view not fully implemented in React demo yet.')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <div style={{ padding: '0.6rem', background: 'var(--accent-gradient)', borderRadius: '12px', color: '#fff', boxShadow: '0 4px 10px rgba(0,117,255,0.3)' }}>
                  <Share2 size={18} />
                </div>
                <div>
                  <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-primary)' }}>Cluster #{cluster.cluster_id}</h3>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: '0.25rem' }}>Identified by GraphSAGE</div>
                </div>
              </div>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                <Users size={16} />
                <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{cluster.size}</span> <span style={{ fontSize: '0.75rem' }}>Nodes</span>
              </div>
              <div style={{ padding: '0.35rem 0.85rem', background: 'var(--danger-gradient)', color: 'white', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 700, boxShadow: '0 4px 10px rgba(227,26,26,0.3)' }}>
                Risk: {cluster.avg_risk.toFixed(3)}
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
