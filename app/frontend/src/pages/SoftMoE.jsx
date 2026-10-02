import { useState, useCallback } from 'react';
import ImageSource from '../components/ImageSource';
import CorruptionControls from '../components/CorruptionControls';
import ImageViewport from '../components/ImageViewport';
import MetricTile from '../components/MetricTile';
import { restoreSoft } from '../lib/api';

const BRANCH_COLORS = ['bg-clean', 'bg-salt', 'bg-blur', 'bg-occlusion'];
const BRANCH_BAR_COLORS = ['bg-clean', 'bg-salt', 'bg-primary', 'bg-error'];

export default function SoftMoE() {
  const [source, setSource] = useState(null);
  const [corruptionParams, setCorruptionParams] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [compareHard, setCompareHard] = useState(false);
  const [includeBranches, setIncludeBranches] = useState(true);

  const handleRestore = useCallback(async () => {
    if (!source) return;
    setLoading(true);
    setError(null);
    try {
      const data = await restoreSoft(source.file, source.sample, {
        ...corruptionParams,
        compare_hard: compareHard,
        include_branches: includeBranches,
      });
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }, [source, corruptionParams, compareHard, includeBranches]);

  const routing = result?.routing;
  const metrics = result?.metrics;
  const mOut = metrics?.output_vs_reference;
  const mIn = metrics?.input_vs_reference;

  return (
    <div className="flex flex-col w-full">
      <div className="w-full bg-surface-container-lowest px-8 py-3 flex items-center justify-between shadow-sm border-b border-outline-variant/20">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-surface-container-low text-primary">
            <span className="material-symbols-outlined text-[20px]">tune</span>
          </div>
          <div>
            <span className="font-semibold text-[16px] text-on-surface">Soft Mixture-of-Experts Restoration</span>
            <p className="text-[12px] text-secondary">Weighted combination of all expert branches via learned gating.</p>
          </div>
        </div>
        {routing && (
          <span className="font-mono text-[11px] text-secondary bg-surface-container-low px-2 py-0.5 rounded border border-outline-variant/30">
            Mode: {routing.mode} · τ={routing.tau ?? '?'}
          </span>
        )}
      </div>

      <div className="w-full p-8 grid grid-cols-12 gap-6 items-start">
        {/* Left */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-5">
          <ImageSource kind="pets" onSelect={setSource} />
          <CorruptionControls onChange={setCorruptionParams} />

          {/* Options */}
          <div className="bg-surface-container-lowest rounded-lg p-4 shadow-sm border border-surface-dim/40 flex flex-col gap-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={includeBranches}
                onChange={(e) => setIncludeBranches(e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <span className="text-[12px] text-on-surface">Show per-branch outputs</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={compareHard}
                onChange={(e) => setCompareHard(e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <span className="text-[12px] text-on-surface">Compare with hard routing</span>
            </label>
          </div>

          <button
            onClick={handleRestore}
            disabled={!source || loading}
            className="w-full py-3 px-4 rounded-lg bg-primary hover:bg-primary-container text-on-primary font-semibold text-[14px] flex items-center justify-center gap-2 shadow-md transition-all active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <span className="material-symbols-outlined text-[20px] animate-spin">refresh</span>
                Running MoE...
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-[20px]">tune</span>
                Execute Soft MoE Restoration
              </>
            )}
          </button>
          {error && (
            <div className="p-3 bg-error-container text-on-error rounded-lg text-[12px]">{error}</div>
          )}
        </div>

        {/* Right */}
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-5">
          {/* Routing weights visualization */}
          {routing && (
            <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 space-y-4">
              <div className="flex items-center justify-between">
                <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">
                  Soft Routing Weights
                </span>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[11px] text-secondary">
                    Entropy: {routing.entropy}
                  </span>
                  <span className={`font-mono text-[11px] px-2 py-0.5 rounded font-medium ${
                    routing.mode === 'dominant' ? 'bg-secondary-container text-on-secondary-container' :
                    routing.mode === 'distributed' ? 'bg-tertiary-container text-on-tertiary' :
                    'bg-surface-container text-on-surface'
                  }`}>
                    {routing.mode}
                  </span>
                </div>
              </div>

              {/* Weight bars */}
              <div className="space-y-3">
                {routing.branch_labels.map((label, i) => {
                  const key = Object.keys(routing.weights)[i];
                  const w = routing.weights_list[i];
                  const dominant = routing.dominant_branch === key;
                  return (
                    <div key={key} className="space-y-1">
                      <div className="flex justify-between text-[12px]">
                        <span className={`flex items-center gap-1.5 ${dominant ? 'font-semibold text-on-surface' : 'text-secondary'}`}>
                          <span className={`w-2.5 h-2.5 rounded-full ${BRANCH_COLORS[i]}`} />
                          {label}
                        </span>
                        <span className={`font-mono text-[11px] ${dominant ? 'font-bold text-primary' : 'text-secondary'}`}>
                          {(w * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2.5 w-full bg-surface-container-low rounded-full overflow-hidden border border-outline-variant/30">
                        <div
                          className={`h-full rounded-full transition-all ${BRANCH_BAR_COLORS[i]}`}
                          style={{ width: `${Math.max(w * 100, 0.5)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Equation */}
              <div className="p-3 bg-surface-container-low rounded font-mono text-[11px] text-on-surface border border-outline-variant/20">
                {routing.equation}
              </div>
            </div>
          )}

          {/* Viewports */}
          <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 space-y-4">
            <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">Visual Inspection</span>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <ImageViewport label="Clean ref" image={result?.images?.reference} footer="Ground Truth" />
              <ImageViewport label="Corrupted input" image={result?.images?.input} footer={mIn ? `PSNR: ${mIn.psnr?.toFixed(2)} dB` : ''} />
              <ImageViewport
                label="MoE output"
                highlight
                image={result?.images?.output}
                footer={mOut ? `PSNR: ${mOut.psnr?.toFixed(2)} dB` : ''}
                footerRight={
                  result?.images?.output && (
                    <a href={result.images.output} download="restored_soft.png" className="p-1 rounded hover:bg-surface-container transition-colors">
                      <span className="material-symbols-outlined text-[15px]">download</span>
                    </a>
                  )
                }
              />
              <ImageViewport label="Error map" tag="|y - ŷ|" image={result?.images?.error_map} footer="Diff Heatmap" />
            </div>
          </div>

          {/* Per-branch outputs */}
          {result?.branch_images && (
            <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 space-y-4">
              <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">Per-Branch Outputs</span>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(result.branch_images).map(([key, img], i) => (
                  <ImageViewport
                    key={key}
                    label={routing?.branch_labels[i] || key}
                    image={img}
                    footer={
                      result.branch_metrics?.[key]
                        ? `PSNR: ${result.branch_metrics[key].psnr?.toFixed(2)} dB`
                        : ''
                    }
                    tag={routing ? `w=${(routing.weights_list[i] * 100).toFixed(1)}%` : ''}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Hard routing comparison */}
          {result?.hard_routing && (
            <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 space-y-3">
              <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">
                Comparison: Hard vs Soft Routing
              </span>
              <div className="grid grid-cols-2 gap-4">
                <ImageViewport
                  label="Hard-routed output"
                  image={result.hard_routing.output}
                  footer={result.hard_routing.metrics?.output_vs_reference ? `PSNR: ${result.hard_routing.metrics.output_vs_reference.psnr?.toFixed(2)} dB` : ''}
                  tag={result.hard_routing.routing?.selected_label}
                />
                <ImageViewport
                  label="Soft MoE output"
                  highlight
                  image={result.images?.output}
                  footer={mOut ? `PSNR: ${mOut.psnr?.toFixed(2)} dB` : ''}
                  tag={`Dominant: ${routing?.dominant_label}`}
                />
              </div>
            </div>
          )}

          {/* Metrics */}
          {metrics && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricTile label="Input PSNR" value={mIn?.psnr?.toFixed(2)} unit="dB" icon="signal_cellular_alt_1_bar" />
              <MetricTile label="MoE PSNR" value={mOut?.psnr?.toFixed(2)} unit="dB" gain={metrics.psnr_gain_db != null ? `${metrics.psnr_gain_db} dB` : undefined} icon="trending_up" highlight />
              <MetricTile label="MoE SSIM" value={mOut?.ssim?.toFixed(3)} gain={metrics.ssim_gain != null ? metrics.ssim_gain.toFixed(4) : undefined} icon="verified" highlight />
              <MetricTile label="Inference" value={result.timing_ms?.inference ?? '?'} unit="ms" icon="speed" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
