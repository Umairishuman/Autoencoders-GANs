# Demo Video Script (~6 minutes)

## Intro (0:00 – 0:30)

"Hi, this is the demo for my Generative AI Assignment 1. The project is a browser-based image restoration and face-to-sketch generation application built with React, Tailwind CSS, and FastAPI, using ONNX Runtime for inference. I'll walk through the Docker startup, all four tasks, and the experiment tracking."

**Screen:** Show the GitHub repo page briefly.

---

## Docker Startup (0:30 – 1:15)

1. Open terminal in the `app/` directory
2. Run: `docker compose up --build`
3. While containers build, explain:
   - "The backend is a FastAPI server loading 7 ONNX models — no PyTorch needed at runtime."
   - "The frontend is React served through Nginx."
4. Show the health logs — models loading, warmup latencies
5. Open browser at `http://localhost`

**Screen:** Terminal → Browser

---

## System & Models Page (1:15 – 1:45)

1. Click "System & Models" in the sidebar
2. Show all 7 models loaded with green "Ready" status
3. Point out model sizes and warmup latencies
4. Click "Open W&B Dashboard" — briefly show experiment tracking page
5. "All training runs were logged to Weights & Biases for reproducibility."

**Screen:** System & Models page → W&B dashboard

---

## Task 1: Universal Restoration (1:45 – 2:45)

1. Click "Universal Restoration" in sidebar
2. Select a sample pet image (e.g. the tabby cat)
3. Apply corruption — choose **Salt & Pepper**, medium severity
4. Click "Corrupt" — show the corrupted image
5. Click "Restore" — show the restored output
6. Point out: "PSNR and SSIM metrics are displayed. The universal autoencoder handles all corruption types without knowing which one was applied."
7. Try another corruption — **Gaussian Blur**, high severity → restore
8. Try **Occlusion** → restore
9. "Best validation PSNR was 24.72 dB across all corruptions."

**Screen:** Universal Restoration workspace

---

## Task 2: Hard-Routed Restoration (2:45 – 3:45)

1. Click "Hard-Routed Restoration" in sidebar
2. Select same pet image
3. Apply salt & pepper corruption → restore
4. Point out: "The classifier predicted the corruption type with 99.88% accuracy, and routed to the salt-and-pepper specialist."
5. Show the classifier probability bars — one class dominates
6. Show the selected expert name and inference time
7. Try with Gaussian blur — show it routes to the blur expert
8. "Each specialist is trained only on its corruption type, achieving 24.86 dB PSNR with predicted routing."

**Screen:** Hard-Routed workspace

---

## Task 3: Soft Mixture-of-Experts (3:45 – 4:45)

1. Click "Soft Mixture-of-Experts" in sidebar
2. Select a pet image, apply occlusion
3. Restore — show the result
4. Point out the gate weights: "The gating network assigns soft weights to all three expert branches. Notice the occlusion expert gets the highest weight."
5. Show branch outputs if visible — "All experts process the image simultaneously, and the final output is a weighted blend."
6. Try with a clean image — show that the identity branch gets high weight
7. "The soft MoE achieved the best PSNR of 25.59 dB — outperforming both the universal and hard-routed approaches."

**Screen:** Soft MoE workspace

---

## Task 4: Face-to-Sketch (4:45 – 5:30)

1. Click "Face-to-Sketch Generator" in sidebar
2. Select a sample face image
3. Choose Style 1 → Generate → show the sketch output
4. Check "Generate all 3 styles" → Generate again
5. Show all three sketches side by side: "Style 1 is clean lines, Style 2 has artistic shading, Style 3 uses bold contours."
6. Try uploading a custom face photo → generate sketch
7. Optionally try the webcam capture → generate
8. Download one of the sketches
9. "The conditional GAN was trained on the FS2K dataset with 2,104 paired images, achieving an FID of 33.29."

**Screen:** Face-to-Sketch workspace

---

## Result Download + Wrap-up (5:30 – 6:00)

1. Download a restored image or sketch using the download button
2. Briefly go back to System & Models — "All models running on CPU via ONNX Runtime."
3. "To summarize: four generative AI tasks — a universal autoencoder, hard-routed specialists with a classifier, a soft mixture-of-experts, and a conditional GAN — all deployed as a single Docker application. Hyperparameters were optimized with Optuna, experiments tracked with Weights & Biases, and models exported to ONNX for efficient inference. Thank you."

**Screen:** Final app overview

---

## Checklist — Must Show

- [x] Application startup (docker compose up)
- [x] Image uploading
- [x] Runtime corruption (all 3 types)
- [x] Universal restoration (Task 1)
- [x] Hard routing + classifier probabilities (Task 2)
- [x] Soft expert weights (Task 3)
- [x] Face-to-sketch generation (Task 4)
- [x] Result downloading
- [x] Experiment tracking records (W&B)
