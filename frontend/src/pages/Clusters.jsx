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

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.5rem' }}>
        {clusters.map((cluster, i) => (
          <Card 
            key={cluster.cluster_id} 
            hover 
            className={`delay-${(i % 5) * 100} animate-fade-in`}
            style={{ cursor: 'pointer' }}
            onClick={() => alert('Cluster detail view not fully implemented in React demo yet. You can build a dedicated route for this!')}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{ padding: '0.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
                  <Share2 size={20} color="var(--accent-primary)" />
                </div>
                <div>
                  <h3 style={{ margin: 0, fontSize: '1.125rem' }}>Cluster #{cluster.cluster_id}</h3>
                </div>
              </div>
              <div style={{ padding: '0.25rem 0.75rem', background: 'rgba(239, 68, 68, 0.1)', color: '#fca5a5', borderRadius: '999px', fontSize: '0.75rem', fontWeight: 600 }}>
                Avg Risk: {cluster.avg_risk.toFixed(3)}
              </div>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
              <Users size={16} />
              <span>{cluster.size} Connected Nodes</span>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
