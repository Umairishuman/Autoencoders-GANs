export default function ImageViewport({ label, tag, image, highlight, badge, footer, footerRight, className = '' }) {
  return (
    <div
      className={`flex flex-col rounded-lg bg-surface-container-low overflow-hidden ${
        highlight ? 'ring-1 ring-primary/40 border border-primary/30' : 'border border-outline-variant/30'
      } ${className}`}
    >
      <div className={`px-3 py-2 flex items-center justify-between ${highlight ? 'bg-primary/10' : 'bg-surface-container'}`}>
        <span className={`font-medium text-[11px] uppercase tracking-wider ${highlight ? 'text-primary font-bold' : 'text-secondary'}`}>
          {label}
        </span>
        {tag && <span className="font-mono text-[10px] text-secondary">{tag}</span>}
      </div>
      <div className="p-3 flex items-center justify-center bg-on-surface/5 min-h-[144px]">
        {image ? (
          <div className="relative">
            <img
              src={image}
              alt={label}
              className="w-32 h-32 object-cover rounded shadow-inner"
              style={{ imageRendering: 'auto' }}
            />
            {badge && (
              <span className="absolute top-1 right-1 bg-primary text-on-primary font-mono text-[9px] px-1.5 py-0.5 rounded shadow">
                {badge}
              </span>
            )}
          </div>
        ) : (
          <div className="w-32 h-32 bg-surface-container rounded flex items-center justify-center">
            <span className="material-symbols-outlined text-outline-variant text-[32px]">image</span>
          </div>
        )}
      </div>
      <div className="px-3 py-1.5 bg-surface-container-lowest flex items-center justify-between text-secondary">
        <span className="font-mono text-[10px]">{footer || ''}</span>
        {footerRight || null}
      </div>
    </div>
  );
}
