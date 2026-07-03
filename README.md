---
title: RRIN Retinal Image Restoration
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# RRIN — Retinal Image Restoration

Upload a degraded retina photograph and receive a restored version from our AI.

## Use the app
Visit the **App** tab above to try it.

## What it restores
- Motion blur and defocus blur
- Uneven illumination and vignetting
- Specular reflections (bright white spots)
- Haze from cataract scattering
- JPEG compression artefacts

## How it works
1. You upload a fundus (retina) photograph
2. The image is sent to our FastAPI backend
3. A U-Net generator with attention gates restores the image
4. You receive a restored PNG in seconds

## Architecture
- Generator: U-Net (4→3 channels) with residual bottleneck and soft attention gates
- Discriminator: Conditional PatchGAN with spectral normalisation
- Loss: L1 + SSIM + VGG perceptual + LSGAN adversarial
- Training data: EyePACS, APTOS 2019, RFMiD, ODIR-5K

⚠️ **Research prototype only** — not for clinical use.
