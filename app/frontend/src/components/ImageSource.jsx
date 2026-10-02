import { useState, useRef, useEffect, useCallback } from 'react';
import { getSamples } from '../lib/api';

export default function ImageSource({ kind = 'pets', onSelect }) {
  const [tab, setTab] = useState('sample');
  const [samples, setSamples] = useState([]);
  const [selected, setSelected] = useState(null);
  const [uploadPreview, setUploadPreview] = useState(null);
  const [uploadName, setUploadName] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef(null);

  useEffect(() => {
    getSamples(kind).then((d) => setSamples(d.samples || [])).catch(() => {});
  }, [kind]);

  const resizeTo128 = useCallback((file) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        canvas.getContext('2d').drawImage(img, 0, 0, 128, 128);
        canvas.toBlob((blob) => {
          resolve(new File([blob], file.name, { type: 'image/png' }));
        }, 'image/png');
        URL.revokeObjectURL(img.src);
      };
      img.src = URL.createObjectURL(file);
    });
  }, []);

  const handleFile = useCallback(
    async (file) => {
      if (!file) return;
      setSelected(null);
      const resized = await resizeTo128(file);
      const previewUrl = URL.createObjectURL(resized);
      setUploadPreview(previewUrl);
      setUploadName(file.name);
      onSelect({ file: resized, sample: null, preview: previewUrl, name: file.name });
    },
    [onSelect, resizeTo128]
  );

  const handleSample = useCallback(
    (s) => {
      setSelected(s.name);
      onSelect({ file: null, sample: s.name, preview: s.url, name: s.name });
    },
    [onSelect]
  );

  return (
    <div className="bg-surface-container-lowest rounded-lg p-5 shadow-sm border border-surface-dim/40 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-primary text-[18px]">photo_library</span>
          <span className="font-semibold text-[16px] text-on-surface">Input Source</span>
        </div>
        <span className="font-mono text-[11px] text-secondary">128×128 RGB</span>
      </div>

      {/* Tabs */}
      <div className="flex rounded-md p-1 bg-surface-container-low gap-1 border border-outline-variant/20">
        <button
          onClick={() => setTab('upload')}
          className={`flex-1 py-1.5 px-3 rounded text-[12px] text-center transition-all flex items-center justify-center gap-1.5 ${
            tab === 'upload'
              ? 'bg-surface-container-lowest text-on-surface font-semibold shadow-sm'
              : 'text-on-surface-variant hover:text-on-surface'
          }`}
        >
          <span className="material-symbols-outlined text-[16px]">cloud_upload</span>
          Upload image
        </button>
        <button
          onClick={() => setTab('sample')}
          className={`flex-1 py-1.5 px-3 rounded text-[12px] text-center transition-all flex items-center justify-center gap-1.5 ${
            tab === 'sample'
              ? 'bg-surface-container-lowest text-on-surface font-semibold shadow-sm'
              : 'text-on-surface-variant hover:text-on-surface'
          }`}
        >
          <span className="material-symbols-outlined text-[16px] text-primary">grid_view</span>
          Use a sample
        </button>
      </div>

      {/* Upload panel */}
      {tab === 'upload' && (
        <>
          <input
            ref={fileRef}
            type="file"
            accept="image/jpeg,image/png,image/webp,image/bmp"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {uploadPreview ? (
            <div className="flex flex-col items-center gap-3 p-4 rounded-lg bg-surface-container-low/50 border border-primary/30">
              <div className="relative w-32 h-32 rounded-lg overflow-hidden ring-2 ring-primary ring-offset-2 ring-offset-surface-container-lowest shadow-md">
                <img src={uploadPreview} alt={uploadName} className="w-full h-full object-cover" />
                <div className="absolute top-1 right-1">
                  <span className="w-5 h-5 rounded-full bg-primary text-on-primary flex items-center justify-center shadow">
                    <span className="material-symbols-outlined text-[14px]">check</span>
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-[11px] text-on-surface font-medium truncate max-w-[140px]">{uploadName}</span>
                <span className="font-mono text-[10px] text-secondary">128×128</span>
              </div>
              <button
                onClick={() => fileRef.current?.click()}
                className="px-3 py-1.5 rounded-md text-[12px] font-medium text-primary hover:bg-primary/10 transition-colors flex items-center gap-1.5 border border-primary/30"
              >
                <span className="material-symbols-outlined text-[16px]">swap_horiz</span>
                Change image
              </button>
            </div>
          ) : (
            <div
              className={`flex flex-col items-center justify-center p-6 rounded-lg border border-dashed transition-all cursor-pointer ${
                dragOver
                  ? 'bg-secondary-container/30 border-primary'
                  : 'bg-surface-container-low/50 border-outline-variant/40 hover:bg-surface-container-low'
              }`}
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFile(e.dataTransfer.files[0]);
              }}
            >
              <div className="w-10 h-10 rounded-full bg-secondary-container/50 flex items-center justify-center text-primary mb-2">
                <span className="material-symbols-outlined text-[22px]">add_photo_alternate</span>
              </div>
              <p className="text-[13px] text-on-surface text-center font-medium">
                Drag & drop image or browse
              </p>
              <p className="font-mono text-[11px] text-secondary mt-1">Resized to 128×128 RGB on ingest</p>
              <span className="mt-3 inline-block px-2 py-0.5 rounded bg-surface-container text-on-surface-variant font-mono text-[11px] border border-outline-variant/30">
                JPG / PNG / WEBP
              </span>
            </div>
          )}
        </>
      )}

      {/* Sample thumbnails */}
      {tab === 'sample' && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between text-secondary">
            <span className="font-medium text-[11px] uppercase tracking-wider">
              Select Benchmark Target
            </span>
            {selected && (
              <span className="font-mono text-[11px]">Selected: {selected}</span>
            )}
          </div>
          <div className="grid grid-cols-3 gap-2.5">
            {samples.map((s) => (
              <button
                key={s.name}
                onClick={() => handleSample(s)}
                className={`group relative aspect-square rounded-lg overflow-hidden transition-all ${
                  selected === s.name
                    ? 'ring-2 ring-primary ring-offset-2 ring-offset-surface-container-lowest'
                    : 'opacity-80 hover:opacity-100 hover:scale-[1.02]'
                }`}
              >
                <img
                  src={s.url}
                  alt={s.name}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
                {selected === s.name && (
                  <div className="absolute inset-0 bg-primary/15 flex items-start justify-end p-1.5">
                    <span className="w-4 h-4 rounded-full bg-primary text-on-primary flex items-center justify-center">
                      <span className="material-symbols-outlined text-[12px] leading-none">check</span>
                    </span>
                  </div>
                )}
                <div className="absolute inset-x-0 bottom-0 bg-on-surface/75 px-1 py-0.5 backdrop-blur-[2px]">
                  <p className="font-mono text-[10px] leading-tight text-surface-bright truncate text-center">
                    {s.name}
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
