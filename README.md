# Ultimate Block Blast Bot 🚀

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange)](https://www.rust-lang.org/)

A professional-grade, high-performance automation suite for the **Block Blast** puzzle game. This bot combines a sophisticated **Rust-accelerated Expectimax solver** with a robust **Multi-Method Cascade Vision pipeline** to achieve world-class scores with machine-like precision.

## 🏆 Performance Milestones
- **Highscore**: **4.6 Million+** (Achieved with **0 revives**)
- **Efficiency**: 100,000 points in under 2 minutes.
- **Precision**: Sub-millisecond solver latency with perfect placement accuracy.

---

## 🛠️ Core Technology Stack

### 🧠 Expectimax Solver (Rust)
The "brain" of the bot is written in **Rust** for maximum throughput. It utilizes an **Expectimax-based Beam Search** algorithm to navigate the massive state space of possible piece combinations.
- **Lookahead**: Optimized search depth to anticipate future piece sets.
- **Weighted Heuristics**: Evaluates board density, hole clusters, combo potential, and "well" depth.
- **Parallel Optimization**: Leverages `rayon` for multi-threaded state evaluation.

### 👁️ Cascade Vision Pipeline
A robust vision system that handles varying screen conditions and animations.
- **LAB Color Multi-Sampling**: Samples background and piece colors in LAB space for high-accuracy segmentation.
- **Cascade Fallback**: Automatically switches between Otsu thresholding, hard thresholding, and Canny edge detection if primary methods fail.
- **Stability Buffer**: Ensures the board is physically stable (animations finished) before computing the next move.

### 🎮 ADB & Input Layer
- **Low-Latency Capture**: Uses Win32 API (`PrintWindow`) for near-instant frame acquisition from `scrcpy`.
- **Precision Input**: Direct ADB `input swipe` commands with calibrated offsets for pixel-perfect placement.
- **Cross-Device Sync**: Included tools to synchronize window dimensions across any machine.

---

## 🚀 Quick Start

### 1. Requirements
- **Python 3.12** (Primary runtime)
- **Rust 1.70+** (For core acceleration)
- **Android Device** with ADB enabled & **scrcpy** installed.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/miles2542/ultimate-block-blast-bot.git
cd ultimate-block-blast-bot

# Install Python dependencies
uv pip install -r requirements.txt

# Build the Rust core extension
maturin develop --manifest-path rust_core/Cargo.toml
```

### 3. Usage
**Step 1: Synchronize Dimensions**
Force `scrcpy` to match the bot's expected calibration (example for 1280x2800 devices):
```bash
scrcpy --window-title="scrcpy" --stay-awake --video-codec=h265 --video-bit-rate=16M --max-fps=60 --video-buffer=0 --window-width=620 --window-height=1346
```

**Step 2: Initialize & Run**
```bash
# Verify calibration (recommended for first run)
python tools/calibrate.py

# Launch the bot
python -m block_blast_bot.main
```

---

## 📂 Project Architecture

```
ultimate-block-blast-bot/
├── src/block_blast_bot/    # Core logic
│   ├── capture/            # High-speed frame acquisition
│   ├── vision/             # Cascade detection & segmentation
│   ├── solver/             # Python-Rust bridge & state logic
│   ├── input/              # ADB mechanic orchestration
│   └── debug/              # Real-time visual overlays (Dear PyGui)
├── rust_core/              # High-performance solver (Rust)
├── tools/                  # Calibration & Diagnostic suites
├── tests/                  # Integrity & Logic test suite
└── docs/                   # Technical guides & performance analysis
```

## 🗺️ Future Roadmap
- [ ] **YOLO Segmentation**: Move towards a calibration-free, device-agnostic vision model.
- [ ] **Improved OOB Experience**: Auto-detection of device resolution and scrcpy config.
- [ ] **Advanced Heuristics**: Neural-network assisted board evaluation for even higher scores.

## ⚖️ License
Distributed under the **Apache License 2.0**. See `LICENSE` for more information.
