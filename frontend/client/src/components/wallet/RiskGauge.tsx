import { useEffect, useState } from 'react';

export function RiskGauge({ score }: { score: number }) {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedScore(score), 100);
    return () => clearTimeout(timer);
  }, [score]);

  const rotation = animatedScore * 180 - 90;

  let scoreColor = 'var(--color-risk-low)';
  if (animatedScore >= 0.5 && animatedScore < 0.8) scoreColor = 'var(--color-risk-medium)';
  if (animatedScore >= 0.8) scoreColor = 'var(--color-risk-high)';

  return (
    <div className="flex flex-col items-center justify-center w-[160px] h-[160px]">
      <svg viewBox="0 0 220 130" className="w-full h-full overflow-visible">
        {/* Low risk arc */}
        <path d="M 20 115 A 90 90 0 0 1 110 25" fill="none" stroke="var(--color-risk-low)" strokeWidth="12" strokeLinecap="round" />
        {/* Medium risk arc */}
        <path d="M 110 25 A 90 90 0 0 1 182.7 62.1" fill="none" stroke="var(--color-risk-medium)" strokeWidth="12" strokeLinecap="round" />
        {/* High risk arc */}
        <path d="M 182.7 62.1 A 90 90 0 0 1 200 115" fill="none" stroke="var(--color-risk-high)" strokeWidth="12" strokeLinecap="round" />
        
        {/* Tick marks */}
        <line x1="20" y1="115" x2="10" y2="115" stroke="var(--color-text-muted)" strokeWidth="2" />
        <line x1="46" y1="51" x2="39" y2="44" stroke="var(--color-text-muted)" strokeWidth="2" />
        <line x1="110" y1="25" x2="110" y2="15" stroke="var(--color-text-muted)" strokeWidth="2" />
        <line x1="174" y1="51" x2="181" y2="44" stroke="var(--color-text-muted)" strokeWidth="2" />
        <line x1="200" y1="115" x2="210" y2="115" stroke="var(--color-text-muted)" strokeWidth="2" />

        {/* Center dot */}
        <circle cx="110" cy="115" r="6" fill="var(--color-text-primary)" />
        
        {/* Needle */}
        <g style={{ transform: `rotate(${rotation}deg)`, transformOrigin: '110px 115px', transition: 'transform var(--duration-gauge) var(--ease-decelerate)' }}>
          <line x1="110" y1="115" x2="110" y2="35" stroke="var(--color-text-primary)" strokeWidth="4" strokeLinecap="round" />
          <polygon points="108,35 112,35 110,25" fill="var(--color-text-primary)" />
        </g>
      </svg>
      <div className="mt-2 text-[var(--text-display)] font-mono leading-none" style={{ color: scoreColor }}>
        {animatedScore.toFixed(2)}
      </div>
    </div>
  );
}
