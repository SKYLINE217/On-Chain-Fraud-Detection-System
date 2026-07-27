import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { SearchInput } from '@/components/shared/SearchInput';
import { RiskGauge } from '@/components/wallet/RiskGauge';
import { LabelBadge } from '@/components/wallet/LabelBadge';
import { ConfidenceBar } from '@/components/wallet/ConfidenceBar';
import { LoadingSpinner } from '@/components/shared/LoadingSpinner';
import { GraphCanvas } from '@/components/graph/GraphCanvas';
import { Shield, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

function ShapBarChart({ data }: { data: any[] }) {
  return (
    <div className="h-[300px] w-full mt-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
          <XAxis type="number" tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }} />
          <YAxis dataKey="feature" type="category" width={100} tick={{ fill: 'var(--color-text-secondary)', fontSize: 12 }} />
          <RechartsTooltip 
            cursor={{ fill: 'var(--color-bg-elevated)' }}
            contentStyle={{ backgroundColor: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)' }}
            formatter={(value: any, name: any, props: any) => [
              `${value > 0 ? '+' : ''}${Number(value).toFixed(3)} (Raw: ${props.payload.feature_value})`, 
              'SHAP Impact'
            ]}
          />
          <ReferenceLine x={0} stroke="var(--color-border)" />
          <Bar dataKey="shap_value" radius={[2, 2, 2, 2]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.shap_value > 0 ? 'var(--color-shap-positive)' : 'var(--color-shap-negative)'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function WalletLookup() {
  const { address } = useParams();
  const navigate = useNavigate();
  
  const [searchInput, setSearchInput] = useState(address || '');
  const [activeAddress, setActiveAddress] = useState(address || '');
  
  const [isLoadingWallet, setIsLoadingWallet] = useState(false);
  const [walletData, setWalletData] = useState<any>(null);

  const [isLoadingExplain, setIsLoadingExplain] = useState(false);
  const [explainData, setExplainData] = useState<any>(null);

  useEffect(() => {
    if (activeAddress) {
      // Mock API call for wallet data
      setIsLoadingWallet(true);
      setWalletData(null);
      setExplainData(null);
      
      setTimeout(() => {
        // Pseudo-randomize based on string length to simulate different results
        const seed = activeAddress.length;
        const isIllicit = seed % 3 === 0;
        
        setWalletData({
          address: activeAddress,
          label: isIllicit ? 'illicit' : 'licit',
          riskScore: isIllicit ? 0.87 : 0.12,
          confidence: isIllicit ? 0.873 : 0.941,
          timeStep: 42 + (seed % 10),
          communityId: 1847 + seed,
          shapFeatures: [
            { feature: 'burst_score', shap_value: isIllicit ? 0.342 : -0.12, feature_value: isIllicit ? 8.7 : 1.2 },
            { feature: 'tx_freq', shap_value: isIllicit ? 0.221 : 0.05, feature_value: isIllicit ? 842.0 : 12.0 },
            { feature: 'value_in', shap_value: isIllicit ? 0.187 : -0.08, feature_value: 14.2 },
            { feature: 'value_out', shap_value: isIllicit ? 0.042 : -0.01, feature_value: 13.9 },
            { feature: 'clustering_c', shap_value: -0.092, feature_value: 0.04 },
          ].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
        });
        setIsLoadingWallet(false);
      }, 800);
    }
  }, [activeAddress]);

  const handleSearch = () => {
    if (searchInput.trim()) {
      navigate(`/wallet/${searchInput.trim()}`);
      setActiveAddress(searchInput.trim());
    }
  };

  const handleExplain = () => {
    setIsLoadingExplain(true);
    setTimeout(() => {
      // Mock Explain Data
      setExplainData({
        subgraph: {
          nodes: [
            { id: activeAddress, risk_score: 0.87, predicted_label: 'illicit', borderColor: '#4D6AF5', borderSize: 2 },
            { id: '0xdef456', risk_score: 0.94, predicted_label: 'illicit' },
            { id: '0x789abc', risk_score: 0.71, predicted_label: 'unknown' },
            { id: '0x123xyz', risk_score: 0.12, predicted_label: 'licit' },
          ],
          edges: [
            { src: activeAddress, dst: '0xdef456', importance_score: 0.94, size: 3 },
            { src: '0x789abc', dst: activeAddress, importance_score: 0.71, size: 2 },
            { src: activeAddress, dst: '0x123xyz', importance_score: 0.1, size: 1 },
          ]
        },
        rationale: `Flagged due to: High burst_score (8.7, +0.342 SHAP); Connected to 2 illicit nodes in community #1847.`
      });
      setIsLoadingExplain(false);
    }, 2000);
  };

  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-slow">
      <div className="space-y-2">
        <h1 className="text-h1 font-bold">Wallet Lookup</h1>
        <p className="text-secondary text-body max-w-2xl">
          Per-wallet risk assessment and feature attribution explainability.
        </p>
      </div>

      {/* Search Area */}
      <SearchInput 
        placeholder="Enter wallet address or txId..."
        value={searchInput}
        onChange={setSearchInput}
        onSubmit={handleSearch}
        isLoading={isLoadingWallet}
      />

      {/* Loading Skeletons */}
      {isLoadingWallet && (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 animate-pulse">
          <div className="md:col-span-4 h-64 bg-surface rounded-md"></div>
          <div className="md:col-span-8 h-64 bg-surface rounded-md"></div>
        </div>
      )}

      {/* Results */}
      {walletData && !isLoadingWallet && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            {/* RiskGauge & Card */}
            <div className="md:col-span-5 flex flex-col gap-6">
              <div className="bg-surface border border-border p-6 rounded-md shadow-card flex items-center justify-between">
                <RiskGauge score={walletData.riskScore} />
                <div className="flex flex-col gap-3 flex-1 ml-6">
                  <div className="text-sm text-secondary">Address</div>
                  <div className="font-mono text-text-primary text-sm bg-sunken px-2 py-1 rounded w-fit">
                    {walletData.address.substring(0, 10)}...
                  </div>
                  
                  <div className="text-sm text-secondary mt-2">Label</div>
                  <div><LabelBadge label={walletData.label} /></div>
                  
                  <div className="text-sm text-secondary mt-2">Confidence</div>
                  <ConfidenceBar confidence={walletData.confidence} label={`${(walletData.confidence * 100).toFixed(1)}%`} />
                  
                  <div className="flex gap-6 mt-2">
                    <div>
                      <div className="text-xs text-secondary">Time step</div>
                      <div className="font-mono mt-1">{walletData.timeStep}</div>
                    </div>
                    <div>
                      <div className="text-xs text-secondary">Community</div>
                      <div className="font-mono mt-1 text-accent cursor-pointer hover:underline">#{walletData.communityId}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* SHAP Features */}
            <div className="md:col-span-7 bg-surface border border-border p-6 rounded-md shadow-card">
              <div className="flex justify-between items-center mb-2">
                <h2 className="text-h2 font-semibold">Feature Attribution (SHAP)</h2>
                <div className="text-muted hover:text-text-primary cursor-help" title="Red bars push the score toward illicit. Blue bars push it toward licit.">
                  <Info className="w-4 h-4" />
                </div>
              </div>
              <ShapBarChart data={walletData.shapFeatures} />
            </div>
          </div>

          {/* Explain Button */}
          {!explainData && (
            <button 
              onClick={handleExplain}
              disabled={isLoadingExplain}
              className="w-full bg-accent text-white h-12 rounded-md font-semibold flex items-center justify-center gap-2 hover:bg-accent/80 transition-colors disabled:opacity-50 shadow-glow"
            >
              {isLoadingExplain ? (
                <>
                  <LoadingSpinner size="sm" />
                  Running GNNExplainer...
                </>
              ) : (
                <>
                  <Shield className="w-5 h-5" />
                  Explain This Wallet (may take 5-15s)
                </>
              )}
            </button>
          )}

          {/* Explanation Panel */}
          {explainData && (
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 animate-in slide-in-from-bottom-4 duration-slow">
              <div className="md:col-span-7 bg-surface border border-border p-1 rounded-md shadow-card">
                <GraphCanvas graphData={explainData.subgraph} height={400} />
              </div>
              <div className="md:col-span-5 bg-surface border border-border p-6 rounded-md shadow-card flex flex-col gap-4">
                <div className="text-caption text-secondary uppercase tracking-widest">Rationale</div>
                <div className="text-sm text-secondary">Explanation Model: GNNExplainer + TreeExplainer</div>
                
                <div className="italic text-body border-l-4 border-danger pl-4 text-text-primary">
                  "{explainData.rationale}"
                </div>

                <div className="mt-4">
                  <div className="text-caption text-secondary uppercase tracking-widest mb-3">Important Nodes</div>
                  <ul className="flex flex-col gap-3">
                    {explainData.subgraph.nodes.filter((n:any) => n.id !== activeAddress).map((node: any) => (
                      <li key={node.id} className="flex items-center justify-between bg-sunken p-2 rounded">
                        <span className="font-mono text-sm">{node.id}</span>
                        <div className="flex items-center gap-3">
                          <span className="text-xs text-secondary">imp: 0.94</span>
                          <LabelBadge label={node.predicted_label} size="sm" />
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
