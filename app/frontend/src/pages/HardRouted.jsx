import { useState, useCallback } from 'react';
import ImageSource from '../components/ImageSource';
import CorruptionControls from '../components/CorruptionControls';
import ImageViewport from '../components/ImageViewport';
import MetricTile from '../components/MetricTile';
import { restoreHard } from '../lib/api';

const CLASS_COLORS = {
  clean: 'bg-clean',
  salt_pepper: 'bg-salt',
  gaussian_blur: 'bg-blur',
  occlusion: 'bg-occlusion',
};

const BAR_COLORS = {
  clean: 'bg-clean',
  salt_pepper: 'bg-salt',
  gaussian_blur: 'bg-primary',
  occlusion: 'bg-error',
};

export default function HardRouted() {
  const [source, setSource] = useState(null);
  const [corruptionParams, setCorruptionParams] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleRestore = useCallback(async () => {
    if (!source) return;
    setLoading(true);
    setError(null);
    try {
      const data = await restoreHard(source.file, source.sample, corruptionParams);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }, [source, corruptionParams]);

  const clf = result?.classifier;
  const routing = result?.routing;
  const metrics = result?.metrics;
  const mOut = metrics?.output_vs_reference;
  const mIn = metrics?.input_vs_reference;

  return (
    <div className="flex flex-col w-full">
      <div className="w-full bg-surface-container-lowest px-8 py-3 flex items-center justify-between shadow-sm border-b border-outline-variant/20">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-surface-container-low text-primary">
            <span className="material-symbols-outlined text-[20px]">alt_route</span>
          </div>
          <div>
            <span className="font-semibold text-[16px] text-on-surface">Hard-Routed Mixture-of-Experts</span>
            <p className="text-[12px] text-secondary">Classifier → identity bypass or single specialist autoencoder.</p>
          </div>
        </div>
        {routing && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-container-low font-mono text-[11px] text-on-surface border border-outline-variant/30">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            ROUTED → {routing.selected_label}
          </div>
        )}
      </div>

      <div className="w-full p-8 grid grid-cols-12 gap-6 items-start">
        {/* Left */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-5">
          <ImageSource kind="pets" onSelect={setSource} />
          <CorruptionControls onChange={setCorruptionParams} />
          <button
            onClick={handleRestore}
            disabled={!source || loading}
            className="w-full py-3 px-4 rounded-lg bg-primary hover:bg-primary-container text-on-primary font-semibold text-[14px] flex items-center justify-center gap-2 shadow-md transition-all active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <span className="material-symbols-outlined text-[20px] animate-spin">refresh</span>
                Running...
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-[20px]">bolt</span>
                Execute Hard-Routing Restoration
              </>
            )}
          </button>
          {error && (
            <div className="p-3 bg-error-container text-on-error rounded-lg text-[12px]">{error}</div>
          )}
        </div>

        {/* Right */}
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-5">
          {/* Routing topology */}
          {clf && (
            <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 space-y-4">
              <div className="flex items-center justify-between">
                <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">
                  MoE Hard-Routed Topology
                </span>
                <span className="font-mono text-[11px] text-secondary">Top-1 Gate: Classifier</span>
              </div>
              {/* Routing flow */}
              <div className="relative bg-surface-container-low rounded-xl p-4 border border-outline-variant/30">
                <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                  {/* Input */}
                  <div className="flex flex-col items-center text-center space-y-1.5 w-28">
                    <div className="w-16 h-16 rounded-lg bg-surface-container-lowest p-1 shadow-sm flex items-center justify-center border border-outline-variant/30 overflow-hidden">
                      {result?.images?.input ? (
                        <img src={result.images.input} alt="input" className="w-full h-full object-cover rounded" />
                      ) : (
                        <span className="material-symbols-outlined text-outline-variant text-[24px]">image</span>
                      )}
                    </div>
                    <span className="font-medium text-[11px] text-on-surface">Corrupted In</span>
                  </div>

                  <span className="material-symbols-outlined text-secondary text-[20px] hidden md:block">arrow_forward</span>

                  {/* Classifier */}
                  <div className="flex flex-col items-center text-center space-y-1.5">
                    <div className="p-3 rounded-lg bg-surface-container-lowest shadow-sm border border-outline-variant/30 flex flex-col items-center">
                      <span className="material-symbols-outlined text-primary text-[22px]">hub</span>
                      <span className="text-[12px] font-semibold text-on-surface">Classifier</span>
                      <span className="font-mono text-[10px] text-primary font-medium">
                        {(clf.confidence * 100).toFixed(1)}% conf
                      </span>
                    </div>
                  </div>

                  <span className="material-symbols-outlined text-secondary text-[20px] hidden md:block">alt_route</span>

                  {/* Expert banks */}
                  <div className="flex flex-col gap-2 w-full md:w-56">
                    {Object.entries(clf.probabilities).map(([name, prob]) => {
                      const selected = clf.predicted === name;
                      return (
                        <div
                          key={name}
                          className={`flex items-center justify-between px-3 py-1.5 rounded-lg transition-all ${
                            selected
                              ? 'bg-surface-container-lowest text-on-surface shadow-md ring-2 ring-primary'
                              : 'bg-surface-container-lowest/70 text-secondary opacity-60 border border-outline-variant/20'
                          }`}
                        >
                          <div className="flex items-center gap-1.5">
                            <span className={`w-2 h-2 rounded-full ${CLASS_COLORS[name] || 'bg-secondary'} ${selected ? 'animate-ping' : ''}`} />
                            <span className="text-[12px]">{name.replace('_', ' ')}</span>
                          </div>
                          <span className={`font-mono text-[11px] ${selected ? 'bg-primary text-on-primary px-1.5 py-0.5 rounded font-medium' : ''}`}>
                            {(prob * 100).toFixed(1)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <span className="material-symbols-outlined text-primary text-[20px] hidden md:block">arrow_forward</span>

                  {/* Output */}
                  <div className="flex flex-col items-center text-center space-y-1.5 w-28">
                    <div className="w-16 h-16 rounded-lg bg-surface-container-lowest p-1 shadow-sm flex items-center justify-center border border-outline-variant/30 ring-2 ring-primary/40 overflow-hidden">
                      {result?.images?.output ? (
                        <img src={result.images.output} alt="output" className="w-full h-full object-cover rounded" />
                      ) : (
                        <span className="material-symbols-outlined text-outline-variant text-[24px]">image</span>
                      )}
                    </div>
                    <span className="font-medium text-[11px] text-on-surface">Restored Out</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Classifier confidence bars + latency */}
          {clf && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="md:col-span-2 bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">Classifier Confidence</span>
                  {routing?.prediction_matches_applied !== null && (
                    <div className="flex items-center gap-1 font-mono text-[11px]">
                      <span className={`material-symbols-outlined text-[15px] ${routing.prediction_matches_applied ? 'text-primary' : 'text-error'}`}>
                        {routing.prediction_matches_applied ? 'verified' : 'error'}
                      </span>
                      <span className={routing.prediction_matches_applied ? 'text-primary' : 'text-error'}>
                        {routing.prediction_matches_applied ? 'Matches Applied' : 'Mismatch!'}
                      </span>
                    </div>
                  )}
                </div>
                <div className="space-y-2">
                  {Object.entries(clf.probabilities).map(([name, prob]) => {
                    const selected = clf.predicted === name;
                    return (
                      <div key={name} className="space-y-1">
                        <div className="flex justify-between text-[12px]">
                          <span className={`flex items-center gap-1.5 ${selected ? 'font-semibold text-on-surface' : 'text-secondary'}`}>
                            <span className={`w-2 h-2 rounded-full ${CLASS_COLORS[name] || 'bg-secondary'}`} />
                            {name.replace('_', ' ')}
                          </span>
                          <span className={`font-mono text-[11px] ${selected ? 'font-bold text-primary' : 'text-secondary'}`}>
                            {(prob * 100).toFixed(1)}%{selected ? ' (Selected)' : ''}
                          </span>
                        </div>
                        <div className={`${selected ? 'h-2' : 'h-1.5'} w-full bg-surface-container-low rounded-full overflow-hidden border border-outline-variant/30`}>
                          <div
                            className={`h-full rounded-full ${BAR_COLORS[name] || 'bg-secondary'}`}
                            style={{ width: `${Math.max(prob * 100, 0.5)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Latency */}
              <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 flex flex-col justify-between space-y-3">
                <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">Latency Profile</span>
                <div className="p-3 bg-surface-container-low rounded-lg space-y-2 font-mono text-[11px] border border-outline-variant/30">
                  <div className="flex justify-between items-center text-secondary">
                    <span>Classifier:</span>
                    <span className="text-on-surface font-semibold">{result.timing_ms?.classifier ?? '?'} ms</span>
                  </div>
                  <div className="flex justify-between items-center text-secondary">
                    <span>Expert:</span>
                    <span className="text-on-surface font-semibold">{result.timing_ms?.expert ?? '?'} ms</span>
                  </div>
                  <div className="pt-2 border-t border-outline-variant/30 flex justify-between items-center text-primary font-bold">
                    <span>Total:</span>
                    <span>{result.timing_ms?.inference ?? '?'} ms</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Viewports */}
          <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 space-y-4">
            <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">High-Acuity Viewports</span>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <ImageViewport label="Clean ref" image={result?.images?.reference} footer="Ground Truth" />
              <ImageViewport label="Input (corrupted)" image={result?.images?.input} footer={mIn ? `PSNR: ${mIn.psnr?.toFixed(2)} dB` : ''} />
              <ImageViewport
                label="Restored output"
                highlight
                image={result?.images?.output}
                badge={routing?.selected_label?.split(' ')[0]}
                footer={mOut ? `PSNR: ${mOut.psnr?.toFixed(2)} dB` : ''}
                footerRight={
                  result?.images?.output && (
                    <a href={result.images.output} download="restored_hard.png" className="p-1 rounded hover:bg-surface-container transition-colors">
                      <span className="material-symbols-outlined text-[15px]">download</span>
                    </a>
                  )
                }
              />
              <ImageViewport label="Residual error" tag="|y - ŷ|" image={result?.images?.error_map} footer="Diff Heatmap" />
            </div>
          </div>

          {/* Metrics strip */}
          {metrics && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <MetricTile
                label="PSNR Reconstruction"
                value={mOut?.psnr?.toFixed(2)}
                unit="dB"
                gain={metrics.psnr_gain_db != null ? `${metrics.psnr_gain_db} dB` : undefined}
                icon="trending_up"
                highlight
              />
              <MetricTile
                label="SSIM Structural"
                value={mOut?.ssim?.toFixed(3)}
                gain={metrics.ssim_gain != null ? metrics.ssim_gain.toFixed(4) : undefined}
                icon="verified"
                highlight
              />
              <MetricTile
                label="Route Confidence"
                value={clf ? `${(clf.confidence * 100).toFixed(1)}%` : '—'}
                icon="hub"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
