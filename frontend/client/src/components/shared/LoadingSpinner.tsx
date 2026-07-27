import { cn } from '@/lib/utils';

export function LoadingSpinner({ size = 'md', label }: { size?: 'sm' | 'md' | 'lg', label?: string }) {
  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-6 h-6 border-2',
    lg: 'w-10 h-10 border-3',
  };

  return (
    <div className="flex flex-col items-center justify-center gap-2">
      <div className={cn(
        "rounded-full border-t-accent border-r-accent border-b-accent border-l-transparent animate-spin",
        sizeClasses[size]
      )} />
      {label && <span className="text-[var(--text-caption)] text-secondary">{label}</span>}
    </div>
  );
}
