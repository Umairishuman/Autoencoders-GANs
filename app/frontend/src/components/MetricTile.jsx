export default function MetricTile({ label, value, unit, gain, gainLabel, icon, highlight }) {
  return (
    <div className={`bg-surface-container-lowest p-4 rounded-lg shadow-sm flex flex-col gap-1 ${
      highlight ? 'border border-primary/30 ring-1 ring-primary/20' : 'border border-surface-dim/40'
    }`}>
      <div className="flex items-center justify-between">
        <span className={`font-medium text-[11px] uppercase ${highlight ? 'text-primary' : 'text-secondary'}`}>
          {label}
        </span>
        {icon && (
          <span className={`material-symbols-outlined text-[16px] ${highlight ? 'text-primary' : 'text-secondary'}`}>
            {icon}
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-1 mt-1">
        <span className="font-mono text-[22px] font-bold text-on-surface">{value ?? '—'}</span>
        {unit && <span className="font-mono text-[11px] text-secondary">{unit}</span>}
      </div>
      {gain !== undefined && gain !== null && (
        <div className="flex items-center gap-1 mt-1">
          <span className="px-1.5 py-0.5 rounded bg-secondary-container text-on-secondary-container font-mono text-[11px] font-semibold">
            {gain > 0 ? '+' : ''}{gain}
          </span>
          {gainLabel && <span className="font-mono text-[11px] text-primary font-medium">{gainLabel}</span>}
        </div>
      )}
    </div>
  );
}
