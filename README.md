# Ghost-Free HDR Imaging using Motion-Aware Deep Learning

Minor Project — B.Tech Computer Engineering  
Zakir Husain College of Engineering & Technology, AMU  
**Kunjal Varshney & Ziya Ali** | Under the guidance of Prof. Saiful Islam

---

## Problem

Digital cameras cannot capture both bright and dark regions of a scene in a single shot. HDR imaging solves this by merging multiple exposures — but any motion between shots introduces **ghosting artifacts**: blurry or doubled regions that degrade the final image.

## Solution

A complete deep learning pipeline that automatically detects and removes ghosting artifacts from multi-exposure HDR images.

---

## Pipeline

```
3 LDR Exposures (Dark, Medium, Bright)
        ↓
1. Optical Flow Registration  →  Align exposures, correct camera shake
        ↓
2. Motion Mask Generation     →  Detect moving pixels (threshold θ = 0.08)
        ↓
3. U-Net (MobileNetV2)        →  Predict motion mask automatically
        ↓
4. Mask-Guided HDR Merge      →  Static: average all 3 | Moving: reference only
        ↓
Ghost-Free HDR Output
```

---

## Results

| Method | Result |
|--------|--------|
| Naive HDR merge (no ghost removal) | Ghost artifacts visible |
| Our mask-guided pipeline | Ghost-free output |
| Kalantari et al. (2017) benchmark | 41.66 dB PSNR |
| Our target range | 35–42 dB PSNR |

---

## Tech Stack

- PyTorch
- U-Net with MobileNetV2 encoder (pretrained on ImageNet)
- Farneback Optical Flow (OpenCV)
- Dataset: [Kalantari HDR Benchmark](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)

---

## Repository Structure

```
├── classical_hdr/       # Traditional HDR methods
├── ml_hdr/              # Deep learning model (U-Net)
├── motion_detection/    # Optical flow & mask generation
├── hybrid_system/       # Full pipeline integration
├── experiments/         # Training experiments & results
├── data/                # Dataset loading utilities
└── utils/               # Helper functions
```

---

## References

- Kalantari & Ramamoorthi, "Deep HDR Imaging of Dynamic Scenes", SIGGRAPH 2017
- Ronneberger et al., "U-Net", MICCAI 2015
- Sandler et al., "MobileNetV2", CVPR 2018
