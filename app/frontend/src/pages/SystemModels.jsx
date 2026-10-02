import { useEffect, useState } from 'react';
import { healthCheck, getModels } from '../lib/api';

export default function SystemModels() {
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([healthCheck(), getModels()])
      .then(([h, m]) => { setHealth(h); setModels(m); })
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="flex flex-col w-full">
      <div className="w-full bg-surface-container-lowest px-8 py-3 flex items-center justify-between shadow-sm border-b border-outline-variant/20">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-surface-container-low text-primary">
            <span className="material-symbols-outlined text-[20px]">memory</span>
          </div>
          <span className="font-semibold text-[16px] text-on-surface">System & Models</span>
        </div>
        <button
          onClick={() => {
            Promise.all([healthCheck(), getModels()])
              .then(([h, m]) => { setHealth(h); setModels(m); setError(null); })
              .catch((e) => setError(e.message));
          }}
          className="px-3 py-1.5 rounded-lg bg-surface-container-low hover:bg-surface-container text-on-surface text-[12px] border border-outline-variant/30 transition-colors flex items-center gap-1.5"
        >
          <span className="material-symbols-outlined text-[16px]">refresh</span>
          Refresh
        </button>
      </div>

      <div className="w-full p-8 space-y-6 max-w-5xl">
        {error && (
          <div className="p-3 bg-error-container text-on-error rounded-lg text-[12px]">{error}</div>
        )}

        {/* Health status */}
        {health && (
          <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 space-y-4">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-[18px]">health_and_safety</span>
              <span className="font-semibold text-[16px] text-on-surface">Runtime Health</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 bg-surface-container-low rounded-lg border border-outline-variant/30 flex flex-col gap-1">
                <span className="text-[11px] uppercase text-secondary font-medium">Status</span>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                  <span className="text-[13px] text-on-surface font-medium">Online</span>
                </div>
              </div>
              <div className="p-3 bg-surface-container-low rounded-lg border border-outline-variant/30 flex flex-col gap-1">
                <span className="text-[11px] uppercase text-secondary font-medium">Uptime</span>
                <span className="text-[13px] text-on-surface font-medium font-mono">
                  {health.uptime_s ? `${Math.floor(health.uptime_s)}s` : '—'}
                </span>
              </div>
              <div className="p-3 bg-surface-container-low rounded-lg border border-outline-variant/30 flex flex-col gap-1">
                <span className="text-[11px] uppercase text-secondary font-medium">ONNX Runtime</span>
                <span className="text-[13px] text-on-surface font-medium font-mono">{health.onnxruntime || '—'}</span>
              </div>
              <div className="p-3 bg-surface-container-low rounded-lg border border-outline-variant/30 flex flex-col gap-1">
                <span className="text-[11px] uppercase text-secondary font-medium">Device</span>
                <span className="text-[13px] text-on-surface font-medium font-mono">{health.device || 'CPU'}</span>
              </div>
            </div>
            {health.providers && (
              <div className="p-3 bg-surface-container-low rounded font-mono text-[11px] text-on-surface border border-outline-variant/20">
                <span className="text-secondary">Providers: </span>
                {health.providers.join(', ')}
              </div>
            )}
          </div>
        )}

        {/* Model registry */}
        {models?.models && (
          <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 space-y-4">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-[18px]">deployed_code</span>
              <span className="font-semibold text-[16px] text-on-surface">ONNX Model Registry</span>
            </div>
            <div className="space-y-3">
              {Object.entries(models.models).map(([role, info]) => (
                <div
                  key={role}
                  className="p-4 bg-surface-container-low rounded-lg border border-outline-variant/30 flex items-center justify-between"
                >
                  <div className="flex items-center gap-3">
                    <span className={`w-2.5 h-2.5 rounded-full ${info.ready ? 'bg-primary' : 'bg-error'}`} />
                    <div>
                      <div className="text-[13px] text-on-surface font-medium">{role}</div>
                      <div className="font-mono text-[11px] text-secondary">{info.file || '—'}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-secondary font-mono text-[11px]">
                    {info.input_shape && <span>Input: {JSON.stringify(info.input_shape)}</span>}
                    {info.warmup_ms != null && <span>Warmup: {info.warmup_ms.toFixed(0)}ms</span>}
                    <span className={`px-2 py-0.5 rounded font-medium ${info.ready ? 'bg-secondary-container text-on-secondary-container' : 'bg-error-container text-error'}`}>
                      {info.ready ? 'Ready' : 'Not loaded'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Sidecars / training info */}
        {models?.sidecars && Object.keys(models.sidecars).length > 0 && (
          <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 space-y-4">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-[18px]">lab_profile</span>
              <span className="font-semibold text-[16px] text-on-surface">Training Sidecars</span>
            </div>
            <div className="space-y-3">
              {Object.entries(models.sidecars).map(([key, info]) => (
                <details key={key} className="bg-surface-container-low rounded-lg border border-outline-variant/30 overflow-hidden">
                  <summary className="p-3 cursor-pointer text-[13px] text-on-surface font-medium hover:bg-surface-container transition-colors">
                    {key}
                  </summary>
                  <div className="p-3 border-t border-outline-variant/20">
                    <pre className="font-mono text-[11px] text-on-surface-variant whitespace-pre-wrap overflow-auto max-h-60">
                      {JSON.stringify(info, null, 2)}
                    </pre>
                  </div>
                </details>
              ))}
            </div>
          </div>
        )}

        {/* W&B link */}
        {models?.experiment_tracking && (
          <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-[18px]">monitoring</span>
              <span className="font-semibold text-[16px] text-on-surface">Experiment Tracking</span>
            </div>
            <a
              href={models.experiment_tracking}
              target="_blank"
              rel="noopener noreferrer"
              className="px-3 py-1.5 rounded-lg bg-primary text-on-primary text-[12px] font-medium shadow-sm flex items-center gap-1.5"
            >
              Open W&B Dashboard
              <span className="material-symbols-outlined text-[14px]">open_in_new</span>
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
