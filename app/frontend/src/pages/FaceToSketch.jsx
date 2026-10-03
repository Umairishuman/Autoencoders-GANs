import { useState, useCallback, useRef } from 'react';
import ImageSource from '../components/ImageSource';
import { generateSketch } from '../lib/api';

const STYLES = [
  { value: '0', label: 'Style 1', desc: 'Clean line sketch' },
  { value: '1', label: 'Style 2', desc: 'Artistic shading' },
  { value: '2', label: 'Style 3', desc: 'Bold contour' },
];

export default function FaceToSketch() {
  const [source, setSource] = useState(null);
  const [style, setStyle] = useState('0');
  const [allStyles, setAllStyles] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [webcamActive, setWebcamActive] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const handleGenerate = useCallback(async () => {
    if (!source) return;
    setLoading(true);
    setError(null);
    try {
      const data = await generateSketch(source.file, source.sample, allStyles ? 'all' : style);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }, [source, style, allStyles]);

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setWebcamActive(true);
      }
    } catch {
      setError('Could not access webcam');
    }
  };

  const stopWebcam = () => {
    videoRef.current?.srcObject?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setWebcamActive(false);
  };

  const captureWebcam = () => {
    const video = videoRef.current;
    if (!video || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const size = 128;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const vw = video.videoWidth || video.width;
    const vh = video.videoHeight || video.height;
    const crop = Math.min(vw, vh);
    const sx = (vw - crop) / 2;
    const sy = (vh - crop) / 2;
    ctx.translate(size, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, sx, sy, crop, crop, 0, 0, size, size);
    canvas.toBlob((blob) => {
      const file = new File([blob], 'webcam.png', { type: 'image/png' });
      setSource({ file, sample: null, preview: URL.createObjectURL(blob), name: 'Webcam capture' });
      stopWebcam();
    }, 'image/png');
  };

  return (
    <div className="flex flex-col w-full">
      <div className="w-full bg-surface-container-lowest px-8 py-3 flex items-center justify-between shadow-sm border-b border-outline-variant/20">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-surface-container-low text-primary">
            <span className="material-symbols-outlined text-[20px]">draw</span>
          </div>
          <div>
            <span className="font-semibold text-[16px] text-on-surface">Face-to-Sketch Generator</span>
            <p className="text-[12px] text-secondary">Style-conditioned conditional GAN (U-Net generator + PatchGAN discriminator)</p>
          </div>
        </div>
        <span className="font-mono text-[11px] text-secondary bg-surface-container-low px-2 py-0.5 rounded border border-outline-variant/30">
          task4_face2sketch_generator.onnx
        </span>
      </div>

      <div className="w-full p-8 grid grid-cols-12 gap-6 items-start">
        {/* Left */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-5">
          <ImageSource kind="faces" onSelect={setSource} />

          {/* Webcam */}
          <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-[18px]">videocam</span>
              <span className="font-semibold text-[16px] text-on-surface">Webcam Capture</span>
            </div>
            {webcamActive ? (
              <div className="flex flex-col items-center gap-3">
                <video ref={videoRef} className="w-48 h-48 rounded-lg object-cover bg-black" style={{ transform: 'scaleX(-1)' }} autoPlay playsInline muted />
                <canvas ref={canvasRef} className="hidden" />
                <div className="flex gap-2">
                  <button
                    onClick={captureWebcam}
                    className="px-4 py-2 rounded-lg bg-primary text-on-primary text-[12px] font-medium shadow-sm flex items-center gap-1"
                  >
                    <span className="material-symbols-outlined text-[16px]">camera</span>
                    Capture
                  </button>
                  <button
                    onClick={stopWebcam}
                    className="px-4 py-2 rounded-lg bg-surface-container-low text-on-surface-variant text-[12px] font-medium border border-outline-variant/30 flex items-center gap-1"
                  >
                    <span className="material-symbols-outlined text-[16px]">close</span>
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={startWebcam}
                className="w-full py-2 rounded-lg bg-surface-container-low hover:bg-surface-container text-on-surface-variant text-[12px] border border-outline-variant/30 transition-colors flex items-center justify-center gap-2"
              >
                <span className="material-symbols-outlined text-[16px]">videocam</span>
                Open Webcam
              </button>
            )}
          </div>

          {/* Style selection */}
          <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-[18px]">palette</span>
              <span className="font-semibold text-[16px] text-on-surface">Sketch Style</span>
            </div>
            <div className="flex flex-col gap-2">
              {STYLES.map((s) => (
                <button
                  key={s.value}
                  onClick={() => { setStyle(s.value); setAllStyles(false); }}
                  disabled={allStyles}
                  className={`flex items-center justify-between p-3 rounded-lg transition-all ${
                    style === s.value && !allStyles
                      ? 'bg-secondary-container/40 text-on-surface shadow-sm ring-1 ring-primary'
                      : 'bg-surface-container-low text-secondary hover:bg-surface-container border border-outline-variant/20'
                  } ${allStyles ? 'opacity-50' : ''}`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`w-3 h-3 rounded-full ${
                      s.value === '0' ? 'bg-primary' : s.value === '1' ? 'bg-tertiary' : 'bg-secondary'
                    }`} />
                    <div className="text-left">
                      <div className="text-[12px] font-medium">{s.label}</div>
                      <div className="text-[11px] text-secondary">{s.desc}</div>
                    </div>
                  </div>
                  {style === s.value && !allStyles && (
                    <span className="material-symbols-outlined text-primary text-[16px]">check</span>
                  )}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 cursor-pointer pt-1">
              <input
                type="checkbox"
                checked={allStyles}
                onChange={(e) => setAllStyles(e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <span className="text-[12px] text-on-surface">Generate all 3 styles</span>
            </label>
          </div>

          <button
            onClick={handleGenerate}
            disabled={!source || loading}
            className="w-full py-3 px-4 rounded-lg bg-primary hover:bg-primary-container text-on-primary font-semibold text-[14px] flex items-center justify-center gap-2 shadow-md transition-all active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <span className="material-symbols-outlined text-[20px] animate-spin">refresh</span>
                Generating Sketch...
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-[20px]">draw</span>
                Generate Sketch
              </>
            )}
          </button>
          {error && (
            <div className="p-3 bg-error-container text-on-error rounded-lg text-[12px]">{error}</div>
          )}
        </div>

        {/* Right */}
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-5">
          {/* Timing */}
          {result && (
            <div className="w-full px-4 py-2.5 rounded-lg bg-surface-container-lowest border border-surface-dim/40 shadow-sm flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
                </span>
                <span className="text-[12px] text-on-surface font-medium">
                  Generated in{' '}
                  <span className="font-mono text-[12px] font-semibold text-primary">
                    {result.timing_ms?.inference ?? '?'} ms
                  </span>
                </span>
                {result.timing_ms?.per_sketch && (
                  <span className="font-mono text-[11px] text-secondary">
                    ({result.timing_ms.per_sketch} ms/sketch)
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Photo + Sketches side by side */}
          <div className="bg-surface-container-lowest rounded-lg p-6 shadow-sm border border-surface-dim/40 space-y-5">
            <div className="flex items-center gap-2.5">
              <span className="material-symbols-outlined text-primary text-[20px]">compare</span>
              <h2 className="font-semibold text-[16px] text-on-surface">Photo → Sketch Results</h2>
            </div>

            <div className={`grid gap-6 ${
              result?.sketches?.length > 1 ? 'grid-cols-1 md:grid-cols-2 xl:grid-cols-4' : 'grid-cols-1 md:grid-cols-2'
            }`}>
              {/* Original photo */}
              <div className="flex flex-col rounded-lg bg-surface-container-low overflow-hidden border border-outline-variant/30">
                <div className="px-3 py-2 bg-surface-container flex items-center justify-between">
                  <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">Original Photo</span>
                </div>
                <div className="p-4 flex items-center justify-center bg-on-surface/5">
                  {result?.images?.photo || source?.preview ? (
                    <img
                      src={result?.images?.photo || source?.preview}
                      alt="Photo"
                      className="w-40 h-40 object-cover rounded shadow-inner"
                    />
                  ) : (
                    <div className="w-40 h-40 bg-surface-container rounded flex items-center justify-center">
                      <span className="material-symbols-outlined text-outline-variant text-[32px]">face</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Sketch outputs */}
              {result?.sketches?.map((sk) => (
                <div key={sk.style} className="flex flex-col rounded-lg bg-surface-container-low overflow-hidden ring-1 ring-primary/40 border border-primary/30">
                  <div className="px-3 py-2 bg-primary/10 flex items-center justify-between">
                    <span className="font-medium text-[11px] uppercase tracking-wider text-primary font-bold">
                      {sk.style_name}
                    </span>
                    <span className="font-mono text-[10px] text-secondary">Generated</span>
                  </div>
                  <div className="p-4 flex items-center justify-center bg-on-surface/5">
                    <img
                      src={sk.image}
                      alt={sk.style_name}
                      className="w-40 h-40 object-cover rounded shadow-inner"
                    />
                  </div>
                  <div className="px-3 py-1.5 bg-surface-container-lowest flex items-center justify-end gap-2">
                    <a
                      href={sk.image}
                      download={`sketch_${sk.style_name.replace(' ', '_').toLowerCase()}.png`}
                      className="p-1 rounded hover:bg-surface-container text-on-surface-variant hover:text-on-surface transition-colors"
                      title="Download sketch"
                    >
                      <span className="material-symbols-outlined text-[15px]">download</span>
                    </a>
                  </div>
                </div>
              ))}

              {/* Placeholder when no result */}
              {!result && (
                <div className="flex flex-col rounded-lg bg-surface-container-low overflow-hidden border border-outline-variant/30">
                  <div className="px-3 py-2 bg-surface-container">
                    <span className="font-medium text-[11px] uppercase tracking-wider text-secondary">Generated Sketch</span>
                  </div>
                  <div className="p-4 flex items-center justify-center bg-on-surface/5">
                    <div className="w-40 h-40 bg-surface-container rounded flex items-center justify-center">
                      <span className="material-symbols-outlined text-outline-variant text-[32px]">draw</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Model info */}
          <div className="bg-surface-container-lowest rounded-lg p-4 shadow-sm border border-surface-dim/40">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-3 bg-surface-container-low rounded-lg border border-outline-variant/30 flex flex-col gap-1">
                <span className="font-medium text-[11px] uppercase text-secondary">Architecture</span>
                <span className="text-[13px] text-on-surface font-medium">U-Net Generator</span>
                <span className="font-mono text-[11px] text-secondary">+ PatchGAN Discriminator</span>
              </div>
              <div className="p-3 bg-surface-container-low rounded-lg border border-outline-variant/30 flex flex-col gap-1">
                <span className="font-medium text-[11px] uppercase text-secondary">Conditioning</span>
                <span className="text-[13px] text-on-surface font-medium">3 Style Embeddings</span>
                <span className="font-mono text-[11px] text-secondary">Learned categorical</span>
              </div>
              <div className="p-3 bg-surface-container-low rounded-lg border border-outline-variant/30 flex flex-col gap-1">
                <span className="font-medium text-[11px] uppercase text-secondary">Dataset</span>
                <span className="text-[13px] text-on-surface font-medium">FS2K</span>
                <span className="font-mono text-[11px] text-secondary">2,104 paired images</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
