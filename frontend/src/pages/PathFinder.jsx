import React, { useState } from 'react';
import { Card, LoadingSpinner } from '../components/Shared';
import { apiClient } from '../api/client';
import { Map, ArrowRight } from 'lucide-react';

export default function PathFinder() {
  const [src, setSrc] = useState('');
  const [dst, setDst] = useState('');
  const [pathResult, setPathResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = (e) => {
    e.preventDefault();
    if (!src || !dst) return;
    
    setLoading(true);
    setError(null);
    setPathResult(null);

    apiClient.getPath(src, dst)
      .then(data => setPathResult(data))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  };

  return (
    <div className="animate-fade-in">
      <h1 className="page-title" style={{ marginBottom: '1.5rem' }}>Transaction Path Finder</h1>
      
      <Card style={{ marginBottom: '2rem' }}>
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end' }}>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Source Address / TxID</label>
            <input 
              type="text" 
              className="search-input" 
              style={{ width: '100%' }}
              value={src} 
              onChange={e => setSrc(e.target.value)} 
              placeholder="e.g. 1A1zP1..."
            />
          </div>
          <div style={{ paddingBottom: '0.75rem', color: 'var(--text-muted)' }}>
            <ArrowRight />
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Destination Address / TxID</label>
            <input 
              type="text" 
              className="search-input" 
              style={{ width: '100%' }}
              value={dst} 
              onChange={e => setDst(e.target.value)} 
              placeholder="e.g. 3J98t1..."
            />
          </div>
          <button type="submit" disabled={loading} style={{
            background: 'var(--accent-primary)',
            color: 'white',
            border: 'none',
            padding: '0.75rem 2rem',
            borderRadius: '999px',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            height: '42px'
          }}>
            {loading ? <LoadingSpinner /> : <><Map size={18} /> Find Path</>}
          </button>
        </form>
      </Card>

      {error && (
        <Card style={{ borderColor: 'var(--danger)', color: 'var(--danger)', marginBottom: '2rem' }}>
          {error}
        </Card>
      )}

      {pathResult && (
        <Card className="animate-fade-in delay-100">
          <h3 style={{ marginBottom: '1.5rem' }}>Path Result</h3>
          
          {pathResult.path_found ? (
            <div>
              <p style={{ color: 'var(--success)', marginBottom: '1rem', fontWeight: 600 }}>
                Found path of length {pathResult.path_length}
              </p>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {pathResult.path_nodes.map((node, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{ width: '2rem', height: '2rem', borderRadius: '50%', background: 'rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.875rem' }}>
                      {i + 1}
                    </div>
                    <div style={{ fontFamily: 'monospace', padding: '0.75rem 1rem', background: 'var(--bg-primary)', borderRadius: '8px', flex: 1, border: '1px solid var(--border-subtle)' }}>
                      {node}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p style={{ color: 'var(--text-secondary)' }}>No path found between these nodes within the maximum depth limit.</p>
          )}
        </Card>
      )}
    </div>
  );
}
