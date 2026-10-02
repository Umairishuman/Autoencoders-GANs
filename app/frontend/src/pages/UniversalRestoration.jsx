import { useState, useCallback } from 'react';
import ImageSource from '../components/ImageSource';
import CorruptionControls from '../components/CorruptionControls';
import ImageViewport from '../components/ImageViewport';
import MetricTile from '../components/MetricTile';
import { restoreUniversal } from '../lib/api';

export default function UniversalRestoration() {
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
      const data = await restoreUniversal(source.file, source.sample, corruptionParams);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }, [source, corruptionParams]);

  const metrics = result?.metrics;
  const mOut = metrics?.output_vs_reference;
  const mIn = metrics?.input_vs_reference;

  return (
    <div className="flex flex-col w-full">
      {/* Sub-header */}
      <div className="w-full bg-surface-container-lowest px-8 py-3 flex items-center justify-between shadow-sm border-b border-outline-variant/20">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-[16px] text-on-surface">Universal Restoration Console</span>
          <span className="font-mono text-[11px] text-secondary bg-surface-container-low px-2 py-0.5 rounded border border-outline-variant/30">
            task1_universal_dae.onnx
          </span>
          <span className="font-medium text-[11px] uppercase tracking-wider text-primary flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
            Runtime Ready
          </span>
        </div>
      </div>

      {/* Main grid */}
      <div className="w-full p-8 grid grid-cols-12 gap-6 items-start">
        {/* Left column */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
          <ImageSource kind="pets" onSelect={setSource} />
          <CorruptionControls onChange={setCorruptionParams} />

          <button
            onClick={handleRestore}
            disabled={!source || loading}
            className="w-full py-3.5 px-4 rounded-lg bg-primary hover:bg-primary-container text-on-primary text-[13px] font-semibold transition-all shadow-md hover:shadow-lg flex items-center justify-center gap-2 active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <span className="material-symbols-outlined text-[20px] animate-spin">refresh</span>
                Running Inference...
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-[20px]">auto_fix_high</span>
                Restore Image
                <span className="ml-2 font-mono text-[11px] px-1.5 py-0.5 rounded bg-white/20 text-on-primary font-normal">
                  Ctrl+Enter
                </span>
              </>
            )}
          </button>

          {error && (
            <div className="p-3 bg-error-container text-on-error rounded-lg text-[12px]">{error}</div>
          )}
        </div>

        {/* Right column */}
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-6">
          {/* Inference notification */}
          {result && (
            <div className="w-full px-4 py-2.5 rounded-lg bg-surface-container-lowest border border-surface-dim/40 shadow-sm flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
                </span>
                <span className="text-[12px] text-on-surface font-medium">
                  Inference complete in{' '}
                  <span className="font-mono text-[12px] font-semibold text-primary">
                    {result.timing_ms?.inference ?? result.timing_ms?.total ?? '?'} ms
                  </span>
                </span>
              </div>
              {metrics?.psnr_gain_db != null && (
                <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-secondary-container text-on-secondary-container font-semibold">
                  PSNR {metrics.psnr_gain_db > 0 ? '+' : ''}{metrics.psnr_gain_db} dB
                </span>
              )}
            </div>
          )}

          {/* Viewport grid */}
          <div className="bg-surface-container-lowest rounded-lg shadow-sm border border-surface-dim/40 p-6 flex flex-col gap-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="material-symbols-outlined text-primary text-[20px]">view_quilt</span>
                <h2 className="font-semibold text-[16px] text-on-surface">Restoration Visual Inspection</h2>
              </div>
              {result && (
                <span className="font-mono text-[11px] bg-surface-container px-2.5 py-1 rounded text-on-surface font-medium border border-outline-variant/30">
                  Inference: {result.timing_ms?.inference ?? '?'} ms
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
              <ImageViewport
                label="Clean reference"
                tag="128×128 RGB"
                image={result?.images?.reference}
                footer="Reference"
                footerRight={<span className="font-mono text-[10px] text-primary font-medium">GT Target</span>}
              />
              <ImageViewport
                label="Corrupted input"
                tag={result?.corruption?.description?.split(',')[0] || ''}
                image={result?.images?.input}
                footer={mIn ? `PSNR: ${mIn.psnr?.toFixed(2) ?? '—'} dB` : ''}
                footerRight={<span className="font-mono text-[10px] text-error font-medium">Degraded</span>}
              />
              <ImageViewport
                label="Restored output"
                highlight
                image={result?.images?.output}
                footer={mOut ? `PSNR: ${mOut.psnr?.toFixed(2) ?? '—'} dB` : ''}
                footerRight={
                  result?.images?.output && (
                    <a
                      href={result.images.output}
                      download="restored.png"
                      className="p-1 rounded hover:bg-surface-container text-on-surface-variant hover:text-on-surface transition-colors"
                      title="Download"
                    >
                      <span className="material-symbols-outlined text-[15px]">download</span>
                    </a>
                  )
                }
              />
              <ImageViewport
                label="Absolute error map"
                tag="|Y - Ŷ|"
                image={result?.images?.error_map}
                footer="Diff Heatmap"
                footerRight={
                  <div className="flex items-center gap-1">
                    <span className="font-mono text-[10px] text-primary font-medium">0.0 — 0.5</span>
                  </div>
                }
              />
            </div>
          </div>

          {/* Metrics */}
          {result && metrics && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricTile label="Input PSNR" value={mIn?.psnr?.toFixed(2)} unit="dB" icon="signal_cellular_alt_1_bar" />
              <MetricTile
                label="Restored PSNR"
                value={mOut?.psnr?.toFixed(2)}
                unit="dB"
                gain={metrics.psnr_gain_db != null ? `${metrics.psnr_gain_db} dB` : undefined}
                gainLabel="gain"
                icon="trending_up"
                highlight
              />
              <MetricTile label="Input SSIM" value={mIn?.ssim?.toFixed(3)} icon="blur_on" />
              <MetricTile
                label="Restored SSIM"
                value={mOut?.ssim?.toFixed(3)}
                gain={metrics.ssim_gain != null ? metrics.ssim_gain.toFixed(4) : undefined}
                icon="verified"
                highlight
              />
            </div>
          )}

          {/* Corruption manifest */}
          {result?.corruption?.applied && (
            <div className="bg-surface-container-lowest rounded-lg p-4 shadow-sm border border-surface-dim/40 flex flex-col gap-2.5">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-salt text-[18px]">terminal</span>
                <span className="font-medium text-[11px] uppercase tracking-wider text-on-surface">
                  Degradation Manifest
                </span>
              </div>
              <div className="p-3 bg-surface-container-low rounded font-mono text-[11px] text-on-surface flex flex-wrap items-center gap-x-4 gap-y-1.5 border border-outline-variant/20">
                <div className="flex items-center gap-1.5">
                  <span className="text-secondary">Type:</span>
                  <span className="font-semibold">{result.corruption.type}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-secondary">Description:</span>
                  <span>{result.corruption.description}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
