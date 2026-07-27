import { MetricCard } from '@/components/shared/MetricCard';
import { TopologyGrid } from '@/components/layout/TopologyGrid';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useEffect, useState } from 'react';

const histogramData = Array.from({ length: 20 }, (_, i) => {
  const bin = (i * 0.05).toFixed(2);
  const center = parseFloat(bin) + 0.025;
  let count = Math.floor(Math.random() * 1000) + 100;
  if (center < 0.2) count += 15000;
  if (center > 0.8) count += 2000;
  return {
    bin,
    center,
    count,
  };
});

function RiskHistogram() {
  return (
    <div className="h-[260px] w-full mt-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={histogramData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <XAxis 
            dataKey="bin" 
            ticks={['0.00', '0.50', '1.00']} 
            tick={{ fill: 'var(--color-text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }} 
            axisLine={false} 
            tickLine={false} 
          />
          <YAxis 
            tickFormatter={(val) => val.toLocaleString()} 
            tick={{ fill: 'var(--color-text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip 
            cursor={{ fill: 'var(--color-bg-elevated)' }}
            contentStyle={{ backgroundColor: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)' }}
            labelStyle={{ color: 'var(--color-text-secondary)' }}
            itemStyle={{ color: 'var(--color-text-primary)' }}
          />
          <Bar dataKey="count" radius={[2, 2, 0, 0]}>
            {histogramData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.center < 0.5 ? 'var(--color-risk-low)' : entry.center < 0.8 ? 'var(--color-risk-medium)' : 'var(--color-risk-high)'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Overview() {
  const [highRiskWidth, setHighRiskWidth] = useState(0);
  const [medRiskWidth, setMedRiskWidth] = useState(0);
  const [lowRiskWidth, setLowRiskWidth] = useState(0);

  useEffect(() => {
    setTimeout(() => {
      setHighRiskWidth(1.4);
      setMedRiskWidth(4.0);
      setLowRiskWidth(94.6);
    }, 100);
  }, []);

  return (
    <div className="flex flex-col gap-8 pb-12 animate-in fade-in duration-slow">
      {}
      <div className="relative pt-8 pb-12 -mx-4 md:-mx-12 px-4 md:px-12 overflow-hidden">
        <TopologyGrid />
        
        <div className="relative z-10 space-y-2 mb-8">
          <h1 className="text-h1 font-bold">System Overview</h1>
          <p className="text-secondary text-body max-w-2xl">
            Real-time insights from the GraphSAGE fraud detection system. Inspecting transaction typologies across the Ethereum network.
          </p>
        </div>

        <div className="relative z-10 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="backdrop-blur-md bg-surface/80 rounded-md">
            <MetricCard label="Total Wallets" value="203,769" />
          </div>
          <div className="backdrop-blur-md bg-surface/80 rounded-md">
            <MetricCard label="High Risk (>80%)" value="2,847" status="danger" />
          </div>
          <div className="backdrop-blur-md bg-surface/80 rounded-md">
            <MetricCard label="Last Scored" value="2 hours ago" subtext="2026-07-27 10:30 UTC" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {}
        <div className="lg:col-span-7 bg-surface border border-border p-6 rounded-md shadow-card">
          <h2 className="text-h2 font-semibold">Risk Distribution</h2>
          <RiskHistogram />
        </div>

        {}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-4">
            <MetricCard label="PR-AUC" value="0.892" />
            <MetricCard label="F1 Score" value="0.781" />
            <MetricCard label="Precision" value="0.914" />
            <MetricCard label="Recall" value="0.682" />
          </div>
          <div className="text-caption text-secondary mt-2 px-1">
            Model: GraphSAGE · Test steps 40–49
          </div>
        </div>
      </div>

      {}
      <div className="bg-surface border border-border p-6 rounded-md shadow-card">
        <h2 className="text-h2 font-semibold mb-6">Risk Tier Breakdown</h2>
        <div className="flex flex-col gap-6 font-mono text-sm">
          <div className="flex items-center gap-4">
            <div className="w-32 text-secondary">High (&gt;0.8)</div>
            <div className="w-32 text-right">2,847 wallets</div>
            <div className="flex-1 h-3 bg-sunken rounded-full overflow-hidden">
              <div className="h-full bg-danger transition-all duration-slow" style={{ width: `${highRiskWidth}%` }} />
            </div>
            <div className="w-16 text-right">1.4%</div>
          </div>
          <div className="flex items-center gap-4">
            <div className="w-32 text-secondary">Med (0.5–0.8)</div>
            <div className="w-32 text-right">8,234 wallets</div>
            <div className="flex-1 h-3 bg-sunken rounded-full overflow-hidden">
              <div className="h-full bg-warning transition-all duration-slow" style={{ width: `${medRiskWidth}%` }} />
            </div>
            <div className="w-16 text-right">4.0%</div>
          </div>
          <div className="flex items-center gap-4">
            <div className="w-32 text-secondary">Low (&lt;0.5)</div>
            <div className="w-32 text-right">192,688 wallets</div>
            <div className="flex-1 h-3 bg-sunken rounded-full overflow-hidden">
              <div className="h-full bg-success transition-all duration-slow" style={{ width: `${lowRiskWidth}%` }} />
            </div>
            <div className="w-16 text-right">94.6%</div>
          </div>
        </div>
      </div>
    </div>
  );
}
