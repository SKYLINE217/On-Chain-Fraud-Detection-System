import { cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  trend?: { direction: 'up' | 'down' | 'flat'; label: string };
  status?: 'success' | 'warning' | 'danger' | 'neutral';
}

export function MetricCard({ label, value, subtext, trend, status = 'neutral' }: MetricCardProps) {
  const statusColors = {
    success: 'text-success',
    warning: 'text-warning',
    danger: 'text-danger',
    neutral: 'text-foreground',
  };

  const TrendIcon = trend?.direction === 'up' ? TrendingUp : trend?.direction === 'down' ? TrendingDown : Minus;
  
  return (
    <div className="bg-surface border border-border rounded-md p-5 flex flex-col gap-2 shadow-card hover:border-accent hover:shadow-glow transition-all duration-fast">
      <div className="text-[var(--text-label)] text-secondary uppercase tracking-widest">{label}</div>
      <div className={cn("text-[var(--text-display)] font-mono leading-none", statusColors[status])}>
        {value}
      </div>
      {subtext && <div className="text-[var(--text-caption)] text-muted mt-1">{subtext}</div>}
      {trend && (
        <div className={cn(
          "flex items-center gap-1 text-xs font-medium mt-1",
          trend.direction === 'up' ? 'text-success' : trend.direction === 'down' ? 'text-danger' : 'text-muted'
        )}>
          <TrendIcon className="w-3 h-3" />
          {trend.label}
        </div>
      )}
    </div>
  );
}
