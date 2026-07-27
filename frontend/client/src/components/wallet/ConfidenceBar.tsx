import { useEffect, useState } from 'react';

export function ConfidenceBar({ confidence, label }: { confidence: number, label?: string }) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setWidth(confidence * 100), 100);
    return () => clearTimeout(timer);
  }, [confidence]);

  return (
    <div className="flex items-center gap-3 w-full">
      <div className="flex-1 h-1.5 bg-surface rounded-full overflow-hidden">
        <div 
          className="h-full rounded-full"
          style={{ 
            width: `${width}%`, 
            background: 'linear-gradient(90deg, var(--color-risk-low), var(--color-risk-medium), var(--color-risk-high))',
            transition: 'width 0.6s cubic-bezier(0, 0, 0.2, 1)'
          }}
        />
      </div>
      {label && <span className="text-xs font-mono font-medium text-text-primary">{label}</span>}
    </div>
  );
}
