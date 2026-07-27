import { Search } from 'lucide-react';
import { LoadingSpinner } from './LoadingSpinner';
import { cn } from '@/lib/utils';

interface SearchInputProps {
  placeholder: string;
  value: string;
  onChange: (val: string) => void;
  onSubmit: () => void;
  isLoading?: boolean;
  error?: string | null;
}

export function SearchInput({ placeholder, value, onChange, onSubmit, isLoading, error }: SearchInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      onSubmit();
    }
  };

  return (
    <div className="w-full max-w-[640px] flex flex-col gap-1">
      <div className={cn(
        "flex items-center gap-2 bg-surface border rounded-md px-3 h-12 shadow-card focus-within:border-accent transition-colors",
        error ? "border-danger" : "border-border"
      )}>
        <Search className="w-5 h-5 text-muted shrink-0" />
        <input 
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="flex-1 bg-transparent border-none outline-none text-text-primary placeholder:text-muted font-mono"
        />
        {isLoading ? (
          <LoadingSpinner size="sm" />
        ) : (
          <button 
            onClick={onSubmit}
            className="flex items-center justify-center bg-accent text-white px-4 h-8 rounded text-sm font-medium hover:bg-accent/80 transition-colors"
          >
            Search
          </button>
        )}
      </div>
      {error && <span className="text-xs text-danger ml-1">{error}</span>}
    </div>
  );
}
