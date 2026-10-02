import { NavLink, Outlet } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { healthCheck } from '../lib/api';

const NAV = [
  { path: '/', label: 'Universal Restoration', icon: 'auto_fix_high' },
  { path: '/hard-routed', label: 'Hard-Routed MoE', icon: 'alt_route' },
  { path: '/soft-moe', label: 'Soft MoE Restoration', icon: 'tune' },
  { path: '/face-sketch', label: 'Face-to-Sketch', icon: 'draw' },
];

const BOTTOM_NAV = [
  { path: '/system', label: 'System & Models', icon: 'memory' },
];

export default function Layout() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    healthCheck().then(setHealth).catch(() => setHealth(null));
    const id = setInterval(() => {
      healthCheck().then(setHealth).catch(() => setHealth(null));
    }, 30000);
    return () => clearInterval(id);
  }, []);

  const online = health && health.ready_count !== undefined
    ? health.ready_count
    : health
    ? Object.values(health.models || {}).filter(Boolean).length
    : 0;

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 h-full w-60 bg-surface-container-lowest border-r border-outline-variant/30 z-50 flex flex-col justify-between select-none">
        <div className="flex flex-col">
          <div className="h-14 px-4 flex items-center justify-between border-b border-outline-variant/20">
            <div className="flex items-center gap-2.5 overflow-hidden">
              <span className="material-symbols-outlined text-primary text-[24px]">biotech</span>
              <span className="font-semibold text-[16px] text-on-surface truncate tracking-tight">
                Restoration Lab
              </span>
            </div>
            <span className="font-mono text-[11px] px-1.5 py-0.5 rounded bg-surface-container text-on-surface-variant font-medium border border-outline-variant/40">
              v1.0
            </span>
          </div>
          <div className="px-4 pt-6 pb-2">
            <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">
              Workspaces
            </span>
          </div>
          <nav className="px-2 space-y-1">
            {NAV.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-[13px] ${
                    isActive
                      ? 'bg-primary text-on-primary font-medium shadow-sm'
                      : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface'
                  }`
                }
              >
                <span className="material-symbols-outlined text-[18px]">{item.icon}</span>
                <span className="truncate">{item.label}</span>
              </NavLink>
            ))}
          </nav>
        </div>

        <div className="p-3 border-t border-outline-variant/20 flex flex-col gap-1">
          <div className="px-2 py-1">
            <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">
              Bench Telemetry
            </span>
          </div>
          <nav className="space-y-1">
            <a
              href={health?.experiment_tracking || '#'}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-between px-3 py-1.5 rounded-lg text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-colors text-[12px]"
            >
              <div className="flex items-center gap-2.5">
                <span className="material-symbols-outlined text-[18px]">monitoring</span>
                <span className="truncate">Experiments (W&B)</span>
              </div>
              <span className="material-symbols-outlined text-[14px] text-secondary">open_in_new</span>
            </a>
            {BOTTOM_NAV.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-2.5 px-3 py-1.5 rounded-lg transition-colors text-[12px] ${
                    isActive
                      ? 'bg-primary text-on-primary font-medium'
                      : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface'
                  }`
                }
              >
                <span className="material-symbols-outlined text-[18px]">{item.icon}</span>
                <span className="truncate">{item.label}</span>
              </NavLink>
            ))}
          </nav>
          <div className="mt-3 pt-3 border-t border-outline-variant/20 px-2 flex items-center justify-between text-secondary">
            <span className="font-mono text-[11px]">ONNX Runtime</span>
            {health ? (
              <span className="material-symbols-outlined text-[16px] text-primary">check_circle</span>
            ) : (
              <span className="material-symbols-outlined text-[16px] text-error">error</span>
            )}
          </div>
        </div>
      </aside>

      {/* Main area */}
      <div className="pl-60 flex-1 flex flex-col min-h-screen">
        {/* Header */}
        <header className="fixed top-0 left-60 right-0 h-14 bg-surface-container-lowest/90 backdrop-blur border-b border-outline-variant/30 z-40 flex items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <span className="material-symbols-outlined text-secondary text-[20px]">biotech</span>
            <span className="font-mono text-[11px] text-secondary tracking-normal">
              Academic AI Bench · Image Restoration
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-surface-container-low border border-outline-variant/40 px-3 py-1 rounded-full">
              <span className="relative flex h-2 w-2">
                {health ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
                  </>
                ) : (
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-error" />
                )}
              </span>
              <span className="font-mono text-[11px] text-on-surface font-medium">
                {health ? `API online · ${online} ONNX models` : 'API offline'}
              </span>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="pt-14 bg-background min-h-screen">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
