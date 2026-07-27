export default function ModelRegistry() {
  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-slow">
      <h1 className="text-2xl font-bold">Model Registry</h1>
      <p className="text-secondary text-body max-w-2xl">
        Manage deployed models and view historical W&B runs.
      </p>
      
      <div className="bg-[var(--color-bg-surface)] border-2 border-[var(--color-accent)] shadow-glow rounded-md p-6 relative">
        <div className="absolute top-4 right-4 bg-[var(--color-accent)] text-white px-2 py-0.5 rounded text-xs font-bold flex items-center gap-1">
          ★ DEPLOYED
        </div>
        <h2 className="text-xl font-bold mb-1">GraphSAGE</h2>
        <div className="font-mono text-sm text-[var(--color-text-secondary)] mb-4">Checkpoint: best_model.pt · 2.3MB</div>
        
        <div className="grid grid-cols-3 gap-4 font-mono text-sm">
          <div>Layers: <span className="text-[var(--color-text-primary)]">3</span></div>
          <div>Hidden dim: <span className="text-[var(--color-text-primary)]">128</span></div>
          <div>Dropout: <span className="text-[var(--color-text-primary)]">0.3</span></div>
          <div>PR-AUC: <span className="text-[var(--color-text-primary)]">0.892</span></div>
          <div>F1: <span className="text-[var(--color-text-primary)]">0.781</span></div>
          <div>Scored: <span className="text-[var(--color-text-primary)]">2026-07-27</span></div>
        </div>
      </div>
    </div>
  );
}
