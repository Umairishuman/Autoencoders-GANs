# Restoration Lab — Generative AI Assignment 1

A browser-based application for multi-corruption image restoration and style-conditioned face-to-sketch generation, built with React + Tailwind CSS (frontend) and FastAPI + ONNX Runtime (backend).

## Repository Structure

```
├── app/
│   ├── AI/
│   │   ├── code files/        # Model architectures (restoration.py, classifier.py, moe.py, sketch_gan.py)
│   │   ├── checkpoints/       # PyTorch checkpoints (task1–task4)
│   │   ├── figures/           # Training curves, Optuna plots, test grids, failure cases
│   │   ├── models/            # Exported ONNX models (task1–task4)
│   │   ├── optuna/            # Optuna SQLite study databases
│   │   ├── results/           # CSV/JSON result tables per task
│   │   └── wandb/             # Weights & Biases run logs
│   ├── backend/               # FastAPI backend (ONNX inference, no PyTorch needed)
│   │   ├── app/               # main.py, pipelines.py, imaging.py, config.py, registry.py
│   │   ├── samples/           # Sample pet & face images for the UI
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── frontend/              # React + Tailwind + Vite
│   │   ├── src/
│   │   │   ├── pages/         # UniversalRestoration, HardRouted, SoftMoE, FaceToSketch
│   │   │   ├── components/    # ImageSource, CorruptionControls, ImageViewport, MetricTile
│   │   │   └── lib/api.js     # Axios API client
│   │   ├── Dockerfile
│   │   └── package.json
│   └── docker-compose.yml     # One-command startup
├── Designs/                   # Google Stitch UI mockups (HTML + screenshots)
├── training_scripts/          # Jupyter notebooks for data preparation & training
├── Tests/                     # Manual test images
└── GenAI_Assignment#1.pdf     # Assignment specification
```

## Prerequisites

- **Docker Desktop** (for containerised deployment), OR
- **Node.js ≥ 18** + **Python ≥ 3.10** (for local dev mode)

### Installing Docker Desktop (Windows)

```powershell
winget install Docker.DockerDesktop
```

After installation, restart your machine and open Docker Desktop. Ensure WSL 2 backend is enabled in Settings → General.

## Quick Start — Docker (Recommended)

This is the one-command deployment required by the assignment.

```bash
cd app
docker compose up --build
```

| Service  | URL                    |
|----------|------------------------|
| Frontend | http://localhost       |
| Backend  | http://localhost:8000  |
| Health   | http://localhost:8000/api/health |

To stop: `Ctrl+C` or `docker compose down`.

## Quick Start — Dev Mode

**Backend** (terminal 1):
```bash
cd app/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend** (terminal 2):
```bash
cd app/frontend
npm install
npm run dev
```

Frontend dev server runs at `http://localhost:5173`. The frontend proxies API calls to port 8000.

## ONNX Models

The trained ONNX models are located in `app/AI/models/`:

| Task | Model File | Size |
|------|-----------|------|
| Task 1 | `task1/task1_universal_dae.onnx` | Universal denoising autoencoder |
| Task 2 | `task2/task2_classifier.onnx` | Corruption classifier |
| Task 2 | `task2/task2_expert_salt_pepper.onnx` | Salt-and-pepper specialist |
| Task 2 | `task2/task2_expert_gaussian_blur.onnx` | Gaussian blur specialist |
| Task 2 | `task2/task2_expert_occlusion.onnx` | Occlusion specialist |
| Task 3 | `task3/task3_soft_moe.onnx` | Soft mixture-of-experts (gate + 3 experts) |
| Task 4 | `task4/task4_face2sketch_generator.onnx` | Face-to-sketch generator |

These files are gitignored due to size. If missing, download them from: *[add your download link here]*

The Docker Compose configuration mounts `app/AI/models/` read-only into the backend container.

## Application Workspaces

The application provides four workspaces accessible via the sidebar:

1. **Universal Restoration** — Upload or select a pet image, optionally apply a corruption (salt-and-pepper, Gaussian blur, or occlusion at low/medium/high severity), and restore it using the Task 1 universal autoencoder.

2. **Hard-Routed Restoration** — The corruption classifier predicts the corruption type, then routes the image to the appropriate specialist autoencoder. Displays classifier probabilities, selected expert, and inference time.

3. **Soft Mixture-of-Experts** — All three expert branches process the image simultaneously with learned routing weights from the gating network. Displays per-branch weights and the blended result.

4. **Face-to-Sketch Generator** — Upload a facial photograph (or use a sample), select Style 1/2/3, and generate a sketch using the conditional GAN generator.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + loaded model list |
| GET | `/api/models` | Model registry with metadata |
| GET | `/api/samples?kind=pets\|faces` | List sample images |
| POST | `/api/corrupt` | Apply corruption to an image |
| POST | `/api/restore/universal` | Task 1 — universal restoration |
| POST | `/api/restore/hard` | Task 2 — hard-routed restoration |
| POST | `/api/restore/soft` | Task 3 — soft MoE restoration |
| POST | `/api/sketch` | Task 4 — face-to-sketch generation |

## Training

All models were trained on Kaggle (GPU). Training notebooks are in `training_scripts/`:

- `00_data_preparation.ipynb` / `00_data_preparation_v2.ipynb` — Oxford-IIIT Pet Dataset download, 80/20 split (seed 42), corruption pipeline, validation/test manifests
- `01_task1_universal_dae.ipynb` / `01_task1_universal_dae_v2.ipynb` — Task 1 training with Optuna
- Additional task notebooks were run as Kaggle notebooks (linked in the report)

Experiment tracking was done with **Weights & Biases**. Local W&B run logs are in `app/AI/wandb/`.

## Hyperparameter Optimization

**Optuna** was used for all four tasks. Study databases are in `app/AI/optuna/`. Summary of trials:

| Task | Study | Completed Trials | Best Trial |
|------|-------|-----------------|------------|
| Task 1 | Universal DAE | 32+ | Trial 32 |
| Task 2 | Classifier | 19+ | Trial 19 |
| Task 2 | Specialists | 12+ | Trial 12 |
| Task 3 | Soft MoE (joint) | 15 | Trial 13 |
| Task 4 | Conditional GAN | 12 (+ 8 pruned) | Trial 13 |

## Key Results

| Metric | Task 1 (Universal) | Task 2 (Hard-Routed) | Task 3 (Soft MoE) |
|--------|-------------------|---------------------|-------------------|
| PSNR (corrupted) | 24.72 dB | 24.86 dB | 25.59 dB |
| SSIM (corrupted) | 0.838 | 0.812 | 0.835 |
| Classifier accuracy | — | 99.88% | — |

| Task 4 | All Styles |
|--------|-----------|
| PSNR | 14.90 dB |
| SSIM | 0.449 |
| FID | 33.29 |

## Tech Stack

- **Frontend**: React 19, Tailwind CSS 4, Vite 8, React Router, Axios
- **Backend**: FastAPI, ONNX Runtime (CPU), Pillow, NumPy
- **Training**: PyTorch, Optuna, Weights & Biases
- **Deployment**: Docker, Docker Compose, Nginx

## License

This project was developed as an individual assignment for the Generative AI course.
