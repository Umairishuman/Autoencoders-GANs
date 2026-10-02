"""Loads every ONNX model once at start-up, warms it up and records its latency."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort

from . import config

log = logging.getLogger("registry")


@dataclass
class LoadedModel:
    role: str
    file: str
    session: ort.InferenceSession | None = None
    error: str | None = None
    size_mb: float = 0.0
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    warmup_ms: float | None = None
    extra_outputs: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.session is not None

    def info(self) -> dict:
        return {"role": self.role, "file": self.file, "loaded": self.ok, "error": self.error, "size_mb": round(self.size_mb, 1),
                "inputs": self.inputs, "outputs": self.outputs, "warmup_latency_ms": self.warmup_ms,
                "extra_outputs": self.extra_outputs}


class Registry:
    def __init__(self):
        self.models: dict[str, LoadedModel] = {}
        self.sidecars: dict[str, dict] = {}
        self.loaded_at = time.time()

    # ------------------------------------------------------------------ loading
    def _options(self) -> ort.SessionOptions:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if config.ORT_THREADS > 0:
            so.intra_op_num_threads = config.ORT_THREADS
        return so

    def _session_with_moe_branches(self, path):
        """Expose the stacked branch outputs of the soft-MoE graph as an extra output named 'branches'.
        This is the trained model's own intermediate tensor, so the per-branch images are exactly what is mixed."""
        import onnx
        m = onnx.load(str(path))
        names = {o for n in m.graph.node for o in n.output}
        if config.MOE_BRANCH_TENSOR not in names:
            raise KeyError(f"tensor {config.MOE_BRANCH_TENSOR} not found in graph")
        m.graph.node.append(onnx.helper.make_node("Identity", [config.MOE_BRANCH_TENSOR], ["branches"], name="expose_branches"))
        m.graph.output.append(onnx.helper.make_tensor_value_info("branches", onnx.TensorProto.FLOAT, None))
        return ort.InferenceSession(m.SerializeToString(), self._options(), providers=["CPUExecutionProvider"])

    def load(self):
        for role, fname in config.MODEL_FILES.items():
            path = config.MODELS_DIR / fname
            lm = LoadedModel(role=role, file=fname)
            if not path.exists():
                lm.error = f"missing file {path}"
                log.warning(lm.error)
                self.models[role] = lm
                continue
            lm.size_mb = path.stat().st_size / 1e6
            try:
                if role == "soft_moe":
                    try:
                        lm.session = self._session_with_moe_branches(path)
                        lm.extra_outputs = ["branches"]
                    except Exception as e:                                       # still usable without branch images
                        log.warning("soft-MoE branch exposure failed (%s); loading plain graph", e)
                        lm.session = ort.InferenceSession(str(path), self._options(), providers=["CPUExecutionProvider"])
                else:
                    lm.session = ort.InferenceSession(str(path), self._options(), providers=["CPUExecutionProvider"])
                lm.inputs = [{"name": i.name, "type": i.type, "shape": i.shape} for i in lm.session.get_inputs()]
                lm.outputs = [{"name": o.name, "shape": o.shape} for o in lm.session.get_outputs()]
            except Exception as e:
                lm.error = repr(e)
                log.exception("failed to load %s", fname)
            self.models[role] = lm
        for key, fname in config.SIDECARS.items():
            p = config.MODELS_DIR / fname
            if p.exists():
                try:
                    self.sidecars[key] = json.loads(p.read_text())
                except Exception as e:
                    log.warning("bad sidecar %s: %s", fname, e)
        if config.WARMUP:
            self.warmup()

    def warmup(self):
        x = np.random.default_rng(0).random((1, 3, config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
        for lm in self.models.values():
            if not lm.ok:
                continue
            feeds = {"photo": x, "style": np.array([0], dtype=np.int64)} if lm.role == "sketch" else {"input": x}
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                lm.session.run(None, feeds)
                ts.append((time.perf_counter() - t0) * 1000)
            lm.warmup_ms = round(float(np.median(ts)), 1)

    # ------------------------------------------------------------------ access
    def get(self, role: str) -> LoadedModel:
        lm = self.models.get(role)
        if lm is None or not lm.ok:
            raise ModelUnavailable(role, lm.error if lm else "unknown model")
        return lm

    def run(self, role: str, feeds: dict, outputs=None):
        lm = self.get(role)
        t0 = time.perf_counter()
        res = lm.session.run(outputs, feeds)
        return res, (time.perf_counter() - t0) * 1000

    def status(self) -> dict:
        n_ok = sum(m.ok for m in self.models.values())
        return {"status": "ok" if n_ok == len(config.MODEL_FILES) else ("degraded" if n_ok else "down"),
                "models_loaded": n_ok, "models_expected": len(config.MODEL_FILES),
                "models": {k: {"loaded": m.ok, "file": m.file, "error": m.error, "warmup_latency_ms": m.warmup_ms}
                           for k, m in self.models.items()}}


class ModelUnavailable(RuntimeError):
    def __init__(self, role, reason):
        super().__init__(f"model '{role}' is not available: {reason}")
        self.role = role


REGISTRY = Registry()
