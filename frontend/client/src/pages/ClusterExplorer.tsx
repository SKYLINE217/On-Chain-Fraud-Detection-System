import { useState, useMemo } from 'react';

const ALL_CLUSTERS = [
  { id: '#1847', size: 234, avg: 0.91, max: 0.99 },
  { id: '#892', size: 187, avg: 0.84, max: 0.97 },
  { id: '#2341', size: 89, avg: 0.72, max: 0.95 },
  { id: '#4910', size: 15, avg: 0.95, max: 1.0 },
  { id: '#112', size: 450, avg: 0.45, max: 0.65 },
  { id: '#773', size: 8, avg: 0.88, max: 0.92 },
  { id: '#9921', size: 1200, avg: 0.12, max: 0.4 },
  { id: '#334', size: 42, avg: 0.65, max: 0.8 },
];

export default function ClusterExplorer() {
  const [minRiskInput, setMinRiskInput] = useState(0.5);
  const [minSizeInput, setMinSizeInput] = useState(10);
  const [sortKey, setSortKey] = useState('avg_desc');
  
  const [filters, setFilters] = useState({ minRisk: 0.5, minSize: 10 });

  const handleApply = () => {
    setFilters({ minRisk: minRiskInput, minSize: minSizeInput });
  };

  const filteredClusters = useMemo(() => {
    const filtered = ALL_CLUSTERS.filter(
      c => c.avg >= filters.minRisk && c.size >= filters.minSize
    );
    return filtered.sort((a, b) => {
      if (sortKey === 'avg_desc') return b.avg - a.avg;
      if (sortKey === 'size_desc') return b.size - a.size;
      if (sortKey === 'id_asc') {
        const idA = parseInt(a.id.replace('#', ''), 10);
        const idB = parseInt(b.id.replace('#', ''), 10);
        return idA - idB;
      }
      return 0;
    });
  }, [filters, sortKey]);

  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-slow">
      <div className="space-y-2">
        <h1 className="text-h1 font-bold">Cluster Explorer</h1>
        <p className="text-secondary text-body max-w-2xl">
          Browse Louvain communities by risk profile.
        </p>
      </div>

      <div className="bg-surface border border-border p-4 rounded-md shadow-card flex flex-wrap items-center gap-6">
        <div className="flex items-center gap-2">
          <label className="text-sm text-secondary">Sort by:</label>
          <select value={sortKey} onChange={e => setSortKey(e.target.value)} className="bg-sunken border border-border rounded px-2 py-1 text-sm outline-none">
            <option value="avg_desc">Avg Risk ▼</option>
            <option value="size_desc">Size ▼</option>
            <option value="id_asc">Community ID</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-secondary">Min Risk:</label>
          <input type="range" min="0" max="1" step="0.05" value={minRiskInput} onChange={e => setMinRiskInput(Number(e.target.value))} className="accent-accent" />
          <span className="font-mono text-sm w-8">{minRiskInput.toFixed(2)}</span>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-sm text-secondary">Min Size:</label>
          <input type="number" value={minSizeInput} onChange={e => setMinSizeInput(Number(e.target.value))} className="bg-sunken border border-border rounded px-2 py-1 text-sm outline-none w-20" />
        </div>
        <button onClick={handleApply} className="bg-accent text-white px-4 py-1.5 rounded text-sm font-medium hover:bg-accent/80 transition-colors ml-auto">
          Apply Filters
        </button>
      </div>

      <div className="bg-surface border border-border rounded-md shadow-card overflow-hidden">
        <table className="w-full text-left text-sm">
          <thead className="bg-elevated/50 text-secondary border-b border-border">
            <tr>
              <th className="p-4 font-medium">Community ID</th>
              <th className="p-4 font-medium">Size</th>
              <th className="p-4 font-medium">Avg Risk</th>
              <th className="p-4 font-medium">Max Risk</th>
              <th className="p-4 font-medium">Risk Bar</th>
              <th className="p-4 font-medium">Action</th>
            </tr>
          </thead>
          <tbody>
            {filteredClusters.length > 0 ? (
              filteredClusters.map(row => (
                <tr key={row.id} className="border-b border-border hover:bg-elevated transition-colors">
                  <td className="p-4 font-mono">{row.id}</td>
                  <td className="p-4">{row.size}</td>
                  <td className="p-4 font-mono">{row.avg}</td>
                  <td className="p-4 font-mono">{row.max}</td>
                  <td className="p-4">
                    <div className="w-full max-w-[160px] h-2 bg-sunken rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${row.avg * 100}%`, background: 'linear-gradient(90deg, var(--color-risk-low), var(--color-risk-medium), var(--color-risk-high))' }} />
                    </div>
                  </td>
                  <td className="p-4">
                    <button 
                      onClick={() => alert(`Navigating to details for community ${row.id}... (WIP)`)}
                      className="text-accent hover:underline text-sm font-medium"
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={6} className="p-8 text-center text-muted">
                  No clusters match the selected filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
        <div className="p-4 text-center text-sm text-secondary border-t border-border flex justify-between items-center">
          <span>Showing {filteredClusters.length} communities.</span>
          <button onClick={() => alert('All available clusters are currently loaded.')} className="text-accent hover:underline ml-1">Load more</button>
        </div>
      </div>
    </div>
  );
}
