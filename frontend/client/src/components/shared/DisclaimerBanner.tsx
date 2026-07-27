import { useState, useEffect } from 'react';
import { AlertTriangle, ChevronRight, ChevronDown } from 'lucide-react';

export default function DisclaimerBanner() {
  const [dismissed, setDismissed] = useState(true); // default true to avoid flash
  const [countdown, setCountdown] = useState(5);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const isDismissed = sessionStorage.getItem('disclaimer_dismissed') === 'true';
    if (!isDismissed) {
      setDismissed(false);
      const timer = setInterval(() => {
        setCountdown((prev) => {
          if (prev <= 1) {
            clearInterval(timer);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
      return () => clearInterval(timer);
    }
  }, []);

  const handleDismiss = () => {
    if (countdown === 0) {
      sessionStorage.setItem('disclaimer_dismissed', 'true');
      setDismissed(true);
    }
  };

  if (dismissed) return null;

  return (
    <div className="fixed bottom-0 left-0 w-full z-50 bg-[#1C2336] border-t-2 border-[var(--color-risk-medium)] shadow-[0_-4px_16px_rgba(0,0,0,0.5)]">
      <div className="max-w-[1440px] mx-auto px-4 md:px-12 py-3">
        {/* Desktop View */}
        <div className="hidden md:flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 text-warning">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            <p className="text-sm font-medium">
              Research/portfolio demonstration only. This system is NOT a certified AML/CFT compliance tool.
            </p>
          </div>
          <button
            onClick={handleDismiss}
            disabled={countdown > 0}
            className={`flex items-center gap-1 text-sm font-medium px-4 py-2 rounded transition-colors whitespace-nowrap ${
              countdown > 0 
                ? 'text-muted opacity-50 cursor-not-allowed' 
                : 'text-text-primary hover:bg-elevated cursor-pointer'
            }`}
          >
            {countdown > 0 ? `Dismiss in ${countdown}s` : 'Dismiss for session'}
            {countdown === 0 && <ChevronRight className="w-4 h-4" />}
          </button>
        </div>

        {/* Mobile View */}
        <div className="md:hidden flex flex-col gap-2">
          <div 
            className="flex items-center justify-between text-warning cursor-pointer"
            onClick={() => setExpanded(!expanded)}
          >
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              <span className="text-sm font-bold">Disclaimer</span>
            </div>
            {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </div>
          
          {expanded && (
            <div className="flex flex-col gap-3 pb-2">
              <p className="text-xs text-text-primary">
                Research/portfolio demonstration only. This system is NOT a certified AML/CFT compliance tool.
              </p>
              <button
                onClick={handleDismiss}
                disabled={countdown > 0}
                className="self-end text-xs font-medium text-muted hover:text-text-primary transition-colors disabled:opacity-50"
              >
                {countdown > 0 ? `Dismiss in ${countdown}s` : 'Dismiss for session'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
