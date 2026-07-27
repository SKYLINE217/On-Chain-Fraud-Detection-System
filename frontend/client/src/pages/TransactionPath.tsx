import { useState } from 'react';
import { GraphCanvas } from '@/components/graph/GraphCanvas';
import { ArrowRight, SearchX } from 'lucide-react';
import { LabelBadge } from '@/components/wallet/LabelBadge';
import { useNavigate } from 'react-router-dom';

export default function TransactionPath() {
  const [src, setSrc] = useState('');
  const [dst, setDst] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [pathData, setPathData] = useState<any>(null);
  const navigate = useNavigate();

  const handleSearch = () => {
    if (!src || !dst) return;
    if (src === dst) {
      setError("Source and destination must be different addresses");
      return;
    }
    setError(null);
    setIsLoading(true);

    setTimeout(() => {
      const found = Math.random() > 0.2; // 80% chance to find a path
      if (found) {
        setPathData({
          hops: 4,
          steps: [
            { from: src, to: '0xABC', label: 'illicit', risk: 0.94 },
            { from: '0xABC', to: '0xDEF', label: 'unknown', risk: 0.71 },
            { from: '0xDEF', to: '0xGHI', label: 'licit', risk: 0.12 },
            { from: '0xGHI', to: dst, label: 'licit', risk: 0.08 },
          ],
          subgraph: {
            nodes: [
              { id: src, risk_score: 0.99, predicted_label: 'illicit', size: 9, borderColor: '#4D6AF5', borderSize: 2 },
              { id: '0xABC', risk_score: 0.94, predicted_label: 'illicit', size: 9, borderColor: '#4D6AF5', borderSize: 2 },
              { id: '0xDEF', risk_score: 0.71, predicted_label: 'unknown', size: 9, borderColor: '#4D6AF5', borderSize: 2 },
              { id: '0xGHI', risk_score: 0.12, predicted_label: 'licit', size: 9, borderColor: '#4D6AF5', borderSize: 2 },
              { id: dst, risk_score: 0.08, predicted_label: 'licit', size: 9, borderColor: '#4D6AF5', borderSize: 2 },
              { id: '0xctx1', risk_score: 0.5, predicted_label: 'unknown', size: 3, color: '#64748B' },
              { id: '0xctx2', risk_score: 0.1, predicted_label: 'licit', size: 3, color: '#64748B' },
            ],
            edges: [
              { src: src, dst: '0xABC', size: 3, color: '#4D6AF5' },
              { src: '0xABC', dst: '0xDEF', size: 3, color: '#4D6AF5' },
              { src: '0xDEF', dst: '0xGHI', size: 3, color: '#4D6AF5' },
              { src: '0xGHI', dst: dst, size: 3, color: '#4D6AF5' },
              { src: src, dst: '0xctx1', size: 0.5, color: '#2A3350' },
              { src: '0xctx1', dst: '0xDEF', size: 0.5, color: '#2A3350' },
            ]
          }
        });
      } else {
        setPathData({ notFound: true });
      }
      setIsLoading(false);
    }, 1500);
  };

  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-slow">
      <div className="space-y-2">
        <h1 className="text-h1 font-bold">Transaction Path</h1>
        <p className="text-secondary text-body max-w-2xl">
          Visualize the shortest path between two addresses up to 10 hops.
        </p>
      </div>

      <div className="bg-surface border border-border p-6 rounded-md shadow-card max-w-2xl">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <label className="text-label text-secondary">Source address</label>
            <input 
              className="bg-sunken border border-border rounded px-3 py-2 font-mono focus:border-accent outline-none" 
              placeholder="0x..." 
              value={src} 
              onChange={e => setSrc(e.target.value)} 
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-label text-secondary">Destination address</label>
            <div className="flex gap-3">
              <input 
                className="flex-1 bg-sunken border border-border rounded px-3 py-2 font-mono focus:border-accent outline-none" 
                placeholder="0x..." 
                value={dst} 
                onChange={e => setDst(e.target.value)} 
                onKeyDown={e => e.key === 'Enter' && handleSearch()}
              />
              <button 
                onClick={handleSearch}
                disabled={isLoading}
                className="bg-accent text-white px-6 rounded font-medium hover:bg-accent/80 transition-colors disabled:opacity-50"
              >
                Find Path
              </button>
            </div>
          </div>
          {error && <div className="text-sm text-danger mt-1">{error}</div>}
        </div>
      </div>

      {isLoading && (
        <div className="h-[400px] bg-surface rounded-md animate-pulse"></div>
      )}

      {pathData && !pathData.notFound && !isLoading && (
        <div className="flex flex-col gap-4 animate-in fade-in duration-slow">
          <div className="text-label text-secondary uppercase tracking-widest">Found path: {pathData.hops} hops</div>
          
          <div className="bg-surface border border-border p-1 rounded-md shadow-card">
            <GraphCanvas graphData={pathData.subgraph} height={400} />
          </div>

          <div className="bg-surface border border-border rounded-md shadow-card overflow-hidden">
            <div className="p-4 border-b border-border bg-elevated/50 font-semibold">Path Steps</div>
            <div className="flex flex-col">
              {pathData.steps.map((step: any, i: number) => (
                <div key={i} className="flex items-center justify-between p-4 border-b border-border last:border-0 hover:bg-elevated transition-colors">
                  <div className="flex items-center gap-4">
                    <span className="text-muted font-medium w-16">Step {i + 1}</span>
                    <span className="font-mono text-sm">{step.from}</span>
                    <ArrowRight className="w-4 h-4 text-muted" />
                    <span className="font-mono text-sm">{step.to}</span>
                  </div>
                  <div className="flex items-center gap-6">
                    <LabelBadge label={step.label} size="sm" />
                    <span className="text-sm text-secondary font-mono w-24 text-right">risk {step.risk.toFixed(2)}</span>
                    <button 
                      onClick={() => navigate(`/wallet/${step.to}`)}
                      className="text-accent hover:text-accent/80 transition-colors p-1"
                    >
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {pathData?.notFound && !isLoading && (
        <div className="h-[400px] bg-surface border border-border rounded-md shadow-card flex flex-col items-center justify-center gap-4 animate-in fade-in text-center">
          <SearchX className="w-12 h-12 text-muted" />
          <div>
            <h3 className="text-lg font-semibold text-text-primary">No path found within 10 hops.</h3>
            <p className="text-secondary mt-1 max-w-md">
              These addresses may not be connected in the transaction graph, or the path exceeds the 10-hop limit.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
