import { AlertCircle } from 'lucide-react';

export function ErrorBanner({ message, onRetry }: { message: string, onRetry?: () => void }) {
  return (
    <div className="w-full flex items-center justify-between gap-4 bg-[var(--color-illicit-badge)] border-l-4 border-[var(--color-risk-high)] p-4 rounded-r-md">
      <div className="flex items-center gap-3">
        <AlertCircle className="w-5 h-5 text-danger shrink-0" />
        <span className="text-sm font-medium text-text-primary">{message}</span>
      </div>
      {onRetry && (
        <button 
          onClick={onRetry}
          className="text-sm font-medium text-danger hover:text-white transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  );
}
