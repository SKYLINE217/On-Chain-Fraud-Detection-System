import { AlertTriangle, CheckCircle, HelpCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

export function LabelBadge({ label, size = 'md' }: { label: 'illicit' | 'licit' | 'unknown', size?: 'sm' | 'md' }) {
  const Icon = label === 'illicit' ? AlertTriangle : label === 'licit' ? CheckCircle : HelpCircle;
  const colorClass = label === 'illicit' ? 'bg-[var(--color-illicit-badge)] text-[var(--color-risk-high)] border-[var(--color-risk-high)]' :
    label === 'licit' ? 'bg-[var(--color-licit-badge)] text-[var(--color-risk-low)] border-[var(--color-risk-low)]' :
    'bg-[var(--color-unknown-badge)] text-[var(--color-risk-unknown)] border-[var(--color-risk-unknown)]';

  return (
    <div className={cn(
      "inline-flex items-center gap-1.5 uppercase tracking-wider font-semibold border border-opacity-20 rounded-full",
      size === 'sm' ? "px-2 py-0.5 text-[0.65rem]" : "px-3 py-1 text-xs",
      colorClass
    )}>
      <Icon className={size === 'sm' ? "w-3 h-3" : "w-4 h-4"} />
      {label}
    </div>
  );
}
