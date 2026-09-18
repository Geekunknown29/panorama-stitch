# Intelligent Panorama Builder - Datasets

This directory contains test image datasets for evaluating the Intelligent Panorama Builder CLI system.

## 1. Sample Overlapping Scene (`data/sample/`)
- `scene1_01.jpg`: Left perspective of a synthetic architectural panorama scene.
- `scene1_02.jpg`: Center perspective overlapping ~40% with `scene1_01.jpg` and ~40% with `scene1_03.jpg`.
- `scene1_03.jpg`: Right perspective overlapping with `scene1_02.jpg`.

**Source / Origin**: Formatted, feature-dense synthetic scenes algorithmically generated specifically for this project (`generate_sample_data.py`). They contain high-frequency edge textures, window grids, text labels, and geometric landmarks designed to produce > 1,000 keypoints and robust RANSAC inlier ratios (~80–90%) for 100% offline reproducibility without external network dependencies.

## 2. Failure Cases (`data/failure_cases/`)
- `unrelated_01.jpg`: High-frequency uniform random noise with zero structural feature correlation.
- `unrelated_02.jpg`: Dark canvas with disconnected text, lacking overlapping keypoint matches with `unrelated_01.jpg`.

**Source / Origin**: Synthetic failure test cases engineered to evaluate robust error handling when feature matching or homography estimation fails.

## Usage
Run the panorama builder CLI on the sample dataset:
```bash
python main.py --input data/sample/scene1_01.jpg data/sample/scene1_02.jpg data/sample/scene1_03.jpg --output outputs/panorama.jpg --debug
```

Or test directory mode:
```bash
python main.py --input-dir data/sample/ --output outputs/panorama.jpg
```

Test failure handling:
```bash
python main.py --input data/failure_cases/unrelated_01.jpg data/failure_cases/unrelated_02.jpg --output outputs/panorama.jpg
```
