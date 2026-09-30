# Task 1: what was wrong with v1 and how v2 fixed it

## Summary

v1 trained well and had good numbers, but it was **not a real autoencoder**. Its skip connections carried more information than the image itself, so the decoder could reconstruct the image without using the compressed latent at all.

v2 compresses the skip connections and adds a hard rule: **everything the decoder receives must be at least 8× smaller than the input image.** v2 loses some quality, and in exchange it satisfies the assignment's "genuine compressed latent" requirement, which can now be demonstrated.

---

## 1. The problem in v1

### What the model looked like
- Encoder: 128×128 → 64 → 32 → 16 → 8, then a 1×1 convolution to the latent (16 channels × 8×8 = **1,024 values**).
- Optuna was allowed to add "limited" skip connections from the 16×16 and 32×32 encoder levels to the decoder. It chose both (`skip_levels = 2`).
- The skips were called "limited" because full-resolution (128×128) and half-resolution (64×64) features were never skipped.

### Why that wasn't enough
"Low resolution" does not mean "little information". What matters is **resolution × number of channels**:

| Code the decoder received (v1) | Shape | Values |
|---|---|---|
| Latent | 16 × 8 × 8 | 1,024 |
| Skip at 16×16 | 384 × 16 × 16 | 98,304 |
| Skip at 32×32 | 192 × 32 × 32 | 196,608 |
| **Total** | | **295,936** |
| Input image | 3 × 128 × 128 | 49,152 |

The decoder received **6× more values than the image contains** ("compression" of 0.17×). Nothing forced information through the latent.

### How we found it: the latent-ablation test
After training, we replaced the latent with zeros and measured the output quality again. If the model really depends on its latent, the output should collapse.

| v1 variant (25 epochs) | Val PSNR | PSNR with latent zeroed | Drop |
|---|---|---|---|
| No skips (pure autoencoder) | 20.6 dB | 7.8 dB | **−12.8 dB**: the latent is essential ✓ |
| 1 skip | 23.8 dB | 23.7 dB | −0.1 dB ✗ |
| 2 skips (**chosen model**) | 26.3 dB | 25.9 dB | **−0.4 dB** ✗ |

For the chosen model, removing the latent cost only 0.4 dB. The decoder was working from the skip features, which amounts to "copying the input through unrestricted skip connections". The assignment explicitly says that does not satisfy the autoencoder requirement.

### Other v1 issues (not related to the model)
- **W&B was off for the whole run.** The `WANDB_API_KEY` secret was attached to notebook 00 but not to the Task 1 notebook, and the notebook continued silently. The results were uploaded afterwards from the saved files (runs tagged `backfilled`).
- **One misleading table row.** A clean input is identical to its target, so its PSNR is infinite; the metric capped it at 100 dB. That 100 dB was averaged into the "overall" row and produced a meaningless "−0.49 dB overall" change.

---

## 2. The fix in v2

### Compressed skip connections
Each skip feature now passes through a **1×1 convolution that squeezes it to 2, 4 or 8 channels** (Optuna chooses). The same dropout as the latent is applied to it. The skips become small compressed codes instead of full feature maps.

### A hard size limit on the whole code
The **total code** (latent + skip codes) must be **at most 6,144 values = 1/8 of the input**. Optuna configurations that break this rule are rejected before training and do not count towards the 30 trials.

The final v2 model:

| Code the decoder receives (v2) | Shape | Values |
|---|---|---|
| Latent | 16 × 8 × 8 | 1,024 |
| Skip code at 16×16 | 4 × 16 × 16 | 1,024 |
| Skip code at 32×32 | 4 × 32 × 32 | 4,096 |
| **Total** | | **6,144 (8× compression)** |

Now the input **cannot** be copied through: every piece of information the decoder uses has to fit into 6,144 values.

Optuna chose a code of exactly 6,144 values, right at the limit. The constraint was active, which shows a real trade-off between compression and quality.

### Other v2 fixes
- **Pre-flight gate.** Before any training, the notebook checks the W&B login (by creating a real test run), the GPU, the dataset, the notebook-00 input, the ONNX tools and the disk space. It stops with instructions if anything is missing.
- **Code-dependence test built in.** After training, the notebook zeroes the latent and the skip codes separately and reports how much each is needed.
- **Table fix.** Clean-input PSNR is left blank (it is infinite) and excluded from the averages. The headline number is the **"all corrupted"** row.

---

## 3. Results: v1 vs v2

### Ablation (same hyperparameters, 20 epochs each)

| Variant | Values to decoder | Compression | Val PSNR | Zero latent | Zero skip codes |
|---|---|---|---|---|---|
| Pure autoencoder | 1,024 | 48× | 21.0 dB | −12.7 dB | – |
| 1 compressed skip | 2,048 | 24× | 21.0 dB | −9.4 dB | −3.6 dB |
| **2 compressed skips (v2)** | **6,144** | **8×** | **24.3 dB** | −2.4 dB | **−15.0 dB** |
| 2 wide skips (v1 design) | 295,936 | 0.17× | 25.8 dB | −1.0 dB | −15.8 dB |

- The v2 design gains **+3.3 dB over the pure autoencoder** while keeping 8× compression.
- Removing all compression (v1) adds only **+1.5 dB more**, and loses the autoencoder property.
- In v2, most information travels through the **32×32 compressed code** (zeroing it costs 15 dB), and the 8×8 latent adds about 2.4 dB of global context. This is a **multi-scale compressed code**, the same idea as hierarchical latent models such as VQ-VAE-2 (Razavi et al., 2019) and Ladder VAEs (Sønderby et al., 2016). The bottleneck that matters is the total code size, which is 8× smaller than the input.

### Test set (3,669 images × 10 conditions)

| | v1 (wide skips) | v2 (8× bottleneck) |
|---|---|---|
| All corrupted inputs, PSNR | 19.3 → 26.5 dB (+7.3) | 19.3 → **24.7 dB (+5.5)** |
| All corrupted inputs, SSIM | 0.63 → 0.88 | 0.63 → **0.84** |
| Clean inputs (output quality) | 29.7 dB / 0.956 | **26.8 dB / 0.897** |
| Salt-and-pepper (PSNR gain) | +13.6 dB | **+10.3 dB** |
| Occlusion (PSNR gain) | +9.0 dB | **+8.2 dB** |
| Gaussian blur (PSNR gain) | −0.7 dB | **−2.2 dB** |

**What this means:**
- The genuine bottleneck costs about **1.8 dB** on corrupted images. That is the price of real compression.
- The model's quality ceiling is now about **26.8 dB**. Salt-and-pepper noise is removed completely (all severities come out near 26.7 dB), and what remains is reconstruction loss.
- **Clean and lightly blurred images get worse**, because their inputs are already better than the ceiling. This is the main motivation for the identity bypass in Task 2 and the identity branch in Task 3.

---

## 4. One-paragraph version for the report

> In the first version (v1), Optuna selected full-width skip connections at the 16×16 and 32×32 levels. A latent-ablation test showed that zeroing the latent reduced validation PSNR by only 0.4 dB. The skips transmitted 295,936 values, six times the size of the input, so the decoder did not need the bottleneck. We therefore squeezed each skip through a 1×1 convolution to a few channels and constrained the total code (latent plus skip codes) to at most one eighth of the input (6,144 values), rejecting non-compliant Optuna configurations before training. The resulting multi-scale code, similar in spirit to hierarchical latents in VQ-VAE-2, keeps +3.3 dB of the skip benefit over a pure autoencoder while guaranteeing 8× compression. The cost is a 1.8 dB lower PSNR on corrupted test images than v1.
