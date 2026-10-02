import { useState, useEffect } from 'react';

const CORRUPTIONS = [
  { key: 'clean', label: 'Clean', color: 'bg-clean' },
  { key: 'salt_pepper', label: 'Salt-and-pepper', color: 'bg-salt' },
  { key: 'gaussian_blur', label: 'Gaussian blur', color: 'bg-blur' },
  { key: 'occlusion', label: 'Occlusion', color: 'bg-occlusion' },
];

const SP_PRESETS = [
  { label: 'Low 0.03', value: 0.03, severity: 'low' },
  { label: 'Medium 0.08', value: 0.08, severity: 'medium' },
  { label: 'High 0.15', value: 0.15, severity: 'high' },
];

const BLUR_PRESETS = [
  { label: '(3, 0.7)', kernel: 3, sigma: 0.7, severity: 'low' },
  { label: '(5, 1.5)', kernel: 5, sigma: 1.5, severity: 'medium' },
  { label: '(7, 2.5)', kernel: 7, sigma: 2.5, severity: 'high' },
];

const OCC_PRESETS = [
  { label: '10% · 1 box', n_boxes: 1, coverage: 0.1, severity: 'low' },
  { label: '20% · 2 box', n_boxes: 2, coverage: 0.2, severity: 'medium' },
  { label: '35% · 3 box', n_boxes: 3, coverage: 0.35, severity: 'high' },
];

export default function CorruptionControls({ onChange, showSkip = true }) {
  const [type, setType] = useState('salt_pepper');
  const [skipCorruption, setSkipCorruption] = useState(false);
  const [seed, setSeed] = useState(42091);
  const [spP, setSpP] = useState(0.08);
  const [blurKernel, setBlurKernel] = useState(5);
  const [blurSigma, setBlurSigma] = useState(1.5);
  const [occBoxes, setOccBoxes] = useState(2);
  const [occCoverage, setOccCoverage] = useState(0.2);

  useEffect(() => {
    if (skipCorruption) {
      onChange({ corruption: 'none', seed });
      return;
    }
    const params = { corruption: type, seed, severity: 'custom' };
    if (type === 'salt_pepper') params.p = spP;
    if (type === 'gaussian_blur') {
      params.kernel_size = blurKernel;
      params.sigma = blurSigma;
    }
    if (type === 'occlusion') {
      params.n_boxes = occBoxes;
      params.coverage = occCoverage;
    }
    if (type === 'clean') params.severity = 'medium';
    onChange(params);
  }, [type, skipCorruption, seed, spP, blurKernel, blurSigma, occBoxes, occCoverage]);

  const reseed = () => setSeed(Math.floor(10000 + Math.random() * 89999));

  return (
    <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-[18px]">grain</span>
          <span className="font-semibold text-[16px] text-on-surface">Apply Corruption</span>
        </div>
      </div>

      {/* Corruption type chips */}
      <div className="grid grid-cols-2 gap-2">
        {CORRUPTIONS.map((c) => (
          <button
            key={c.key}
            onClick={() => setType(c.key)}
            disabled={skipCorruption}
            className={`px-3 py-2 rounded-lg flex items-center justify-between transition-all ${
              type === c.key && !skipCorruption
                ? 'bg-surface-container text-on-surface font-semibold shadow-sm ring-1 ring-primary'
                : 'bg-surface-container-low text-secondary hover:bg-surface-container border border-outline-variant/20'
            } ${skipCorruption ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <div className="flex items-center gap-2">
              <span className={`w-2.5 h-2.5 rounded-full ${c.color}`} />
              <span className="text-[12px]">{c.label}</span>
            </div>
            {type === c.key && !skipCorruption && (
              <span className="material-symbols-outlined text-primary text-[16px]">check_circle</span>
            )}
          </button>
        ))}
      </div>

      {/* Salt-and-pepper controls */}
      {type === 'salt_pepper' && !skipCorruption && (
        <div className="p-3.5 bg-surface-container-low rounded-lg flex flex-col gap-3 border border-outline-variant/25">
          <div className="flex items-center justify-between">
            <span className="text-[12px] font-medium text-on-surface">
              Noise probability <span className="font-mono italic text-secondary">p</span>
            </span>
            <span className="font-mono text-[12px] text-on-surface bg-surface-container-lowest px-2 py-0.5 rounded shadow-sm border border-outline-variant/30">
              {spP.toFixed(3)}
            </span>
          </div>
          <input
            type="range"
            min={0.02}
            max={0.15}
            step={0.01}
            value={spP}
            onChange={(e) => setSpP(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-surface-container-highest rounded-lg cursor-pointer"
          />
          <div className="flex justify-between font-mono text-[10px] text-secondary">
            <span>0.02 (Subtle)</span>
            <span>0.08 (Standard)</span>
            <span>0.15 (Severe)</span>
          </div>
          <div className="flex items-center gap-2 pt-1">
            <span className="text-[11px] text-secondary font-medium">Presets:</span>
            {SP_PRESETS.map((p) => (
              <button
                key={p.value}
                onClick={() => setSpP(p.value)}
                className={`px-2 py-0.5 rounded font-mono text-[11px] transition-colors ${
                  Math.abs(spP - p.value) < 0.005
                    ? 'bg-primary text-on-primary shadow-sm'
                    : 'bg-surface-container hover:bg-surface-container-high text-on-surface-variant border border-outline-variant/30'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Gaussian blur controls */}
      {type === 'gaussian_blur' && !skipCorruption && (
        <div className="p-3.5 bg-surface-container-low rounded-lg flex flex-col gap-3 border border-outline-variant/25">
          <div className="flex items-center justify-between">
            <span className="text-[12px] text-secondary">Kernel Dimensions (k)</span>
            <div className="flex p-0.5 bg-surface-container-lowest rounded-md border border-outline-variant/30">
              {[3, 5, 7].map((k) => (
                <button
                  key={k}
                  onClick={() => setBlurKernel(k)}
                  className={`px-2.5 py-0.5 font-mono text-[11px] rounded ${
                    blurKernel === k
                      ? 'bg-primary text-on-primary font-medium shadow-sm'
                      : 'text-secondary hover:text-on-surface'
                  }`}
                >
                  {k}×{k}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[12px] text-secondary">Spread Parameter (σ)</span>
            <span className="font-mono text-[12px] px-2 py-0.5 rounded bg-surface-container-lowest text-on-surface font-semibold shadow-sm border border-outline-variant/30">
              {blurSigma.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0.5}
            max={2.5}
            step={0.1}
            value={blurSigma}
            onChange={(e) => setBlurSigma(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-surface-container-highest rounded-lg cursor-pointer"
          />
          <div className="flex items-center gap-1.5 pt-1">
            {BLUR_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => { setBlurKernel(p.kernel); setBlurSigma(p.sigma); }}
                className={`px-2 py-0.5 rounded-full font-mono text-[11px] shadow-sm ${
                  blurKernel === p.kernel && Math.abs(blurSigma - p.sigma) < 0.05
                    ? 'bg-primary text-on-primary font-medium'
                    : 'bg-surface-container-lowest text-secondary hover:text-on-surface border border-outline-variant/30'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Occlusion controls */}
      {type === 'occlusion' && !skipCorruption && (
        <div className="p-3.5 bg-surface-container-low rounded-lg flex flex-col gap-3 border border-outline-variant/25">
          <div className="flex items-center justify-between">
            <span className="text-[12px] text-secondary">Box count</span>
            <div className="flex p-0.5 bg-surface-container-lowest rounded-md border border-outline-variant/30">
              {[1, 2, 3].map((n) => (
                <button
                  key={n}
                  onClick={() => setOccBoxes(n)}
                  className={`px-2.5 py-0.5 font-mono text-[11px] rounded ${
                    occBoxes === n
                      ? 'bg-primary text-on-primary font-medium shadow-sm'
                      : 'text-secondary hover:text-on-surface'
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[12px] text-secondary">Coverage</span>
            <span className="font-mono text-[12px] px-2 py-0.5 rounded bg-surface-container-lowest text-on-surface font-semibold shadow-sm border border-outline-variant/30">
              {(occCoverage * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min={0.1}
            max={0.35}
            step={0.05}
            value={occCoverage}
            onChange={(e) => setOccCoverage(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-surface-container-highest rounded-lg cursor-pointer"
          />
          <div className="flex items-center gap-1.5 pt-1">
            {OCC_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => { setOccBoxes(p.n_boxes); setOccCoverage(p.coverage); }}
                className={`px-2 py-0.5 rounded-full font-mono text-[11px] shadow-sm ${
                  occBoxes === p.n_boxes && Math.abs(occCoverage - p.coverage) < 0.01
                    ? 'bg-primary text-on-primary font-medium'
                    : 'bg-surface-container-lowest text-secondary hover:text-on-surface border border-outline-variant/30'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Seed */}
      <div className="flex items-center justify-between pt-1">
        <label className="text-[12px] text-on-surface-variant font-medium">Random seed</label>
        <div className="flex items-center gap-1.5">
          <div className="px-2.5 py-1 rounded bg-surface-container-low border border-outline-variant/30 font-mono text-[12px] text-on-surface font-medium flex items-center gap-1">
            <span className="text-secondary select-none">#</span>
            <span>{seed}</span>
          </div>
          <button
            onClick={reseed}
            className="p-1.5 rounded bg-surface-container-low hover:bg-surface-container text-on-surface-variant hover:text-on-surface border border-outline-variant/30 transition-colors"
            title="Re-roll seed"
          >
            <span className="material-symbols-outlined text-[16px]">casino</span>
          </button>
        </div>
      </div>

      {/* Skip corruption toggle */}
      {showSkip && (
        <div className="pt-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="skip-corruption"
              checked={skipCorruption}
              onChange={(e) => setSkipCorruption(e.target.checked)}
              className="w-4 h-4 rounded"
            />
            <label htmlFor="skip-corruption" className="text-[12px] text-on-surface cursor-pointer select-none">
              Image is already corrupted
            </label>
          </div>
          <span
            className="material-symbols-outlined text-[16px] text-secondary cursor-help"
            title="Bypasses synthetic degradation and runs denoising directly on the uploaded image"
          >
            help_outline
          </span>
        </div>
      )}
    </div>
  );
}
