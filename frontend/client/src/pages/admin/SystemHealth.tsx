import { MetricCard } from '@/components/shared/MetricCard';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from 'recharts';
import { useState } from 'react';

const latencyData = Array.from({ length: 20 }, (_, i) => ({
  time: i,
  wallet: Math.random() * 200 + 100,
  subgraph: Math.random() * 500 + 200,
  explain: Math.random() * 3000 + 1000,
}));

export default function SystemHealth() {
  const [isFlushed, setIsFlushed] = useState(false);

  const handleFlush = () => {
    setIsFlushed(true);
    setTimeout(() => setIsFlushed(false), 2000);
  };

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-slow">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">System Health</h1>
        <div className="text-sm text-[var(--color-text-secondary)] flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[var(--color-text-success)] animate-pulse" />
          Auto-refreshing · 28s
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard label="FastAPI" value="HEALTHY" status="success" subtext="p95: 234ms | Uptime: 99%" />
        <MetricCard label="Neo4j" value="HEALTHY" status="success" subtext="Nodes: 203,769 | Edges: 234,355" />
        <MetricCard label="Redis" value="HEALTHY" status="success" subtext="Hit rate: 73% | Keys: 8,423" />
        <MetricCard label="BFF" value="UP" status="success" subtext="v1.0.0" />
      </div>

      <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border)] rounded-md p-6 shadow-card">
        <h2 className="text-lg font-semibold mb-4">API Latency (last 100 requests)</h2>
        <div className="h-[200px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={latencyData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <XAxis dataKey="time" hide />
              <YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-md)' }}
                labelStyle={{ color: 'var(--color-text-primary)' }}
              />
              <Legend verticalAlign="bottom" height={36}/>
              <ReferenceLine y={5000} label={{ position: 'top', value: 'Target 5s', fill: 'var(--color-text-muted)', fontSize: 11 }} stroke="var(--color-border)" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="wallet" stroke="#22C55E" dot={false} strokeWidth={2} name="/wallet" />
              <Line type="monotone" dataKey="subgraph" stroke="#3B82F6" dot={false} strokeWidth={2} name="/subgraph" />
              <Line type="monotone" dataKey="explain" stroke="#F59E0B" dot={false} strokeWidth={2} name="/explain" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border)] rounded-md p-6 shadow-card flex items-center justify-between">
        <div className="flex flex-col gap-1">
          <h2 className="text-lg font-semibold">Redis Cache</h2>
          <p className="text-sm text-[var(--color-text-secondary)]">Hit rate: 73.2% &nbsp; | &nbsp; Total keys: 8,423 &nbsp; | &nbsp; Memory: 42MB / 256MB</p>
          <div className="w-64 h-2 bg-[var(--color-bg-sunken)] rounded-full overflow-hidden mt-2">
            <div className="h-full bg-[var(--color-accent)] transition-all duration-500" style={{ width: isFlushed ? '0%' : '16%' }} />
          </div>
        </div>
        <button onClick={handleFlush} className="px-4 py-2 bg-transparent border border-[var(--color-text-danger)] text-[var(--color-text-danger)] rounded font-medium hover:bg-[var(--color-text-danger)]/10 transition-colors">
          {isFlushed ? 'Flushed!' : 'Flush Cache'}
        </button>
      </div>
    </div>
  );
}
