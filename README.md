# Intelligent Panorama Builder

> **Academic Computer Vision Project**  
> A production-quality, modular, and reproducible command-line application for automatic image stitching using classical computer vision algorithms (SIFT/ORB, KNN matching with Lowe's ratio test, RANSAC homography estimation, perspective warping, distance-transform feathered blending, and quantitative evaluation).

---

## 1. Project Overview
The **Intelligent Panorama Builder** is a self-contained Python command-line utility designed for academic evaluation in Computer Vision courses. It accepts multiple overlapping images or an image directory and automatically generates a high-resolution, seamless panoramic image alongside structured quantitative metrics (`metrics.json`).

The system exposes every algorithmic phase of the computer vision pipeline cleanly in code, without concealing logic behind black-box high-level convenience wrappers.

## Quick Start

These steps are sufficient for a clean installation by a first-time evaluator.

1. Install Python 3.10 or newer and make sure `python` is available in a terminal.
2. Clone the repository and enter its directory:

    ```bash
    git clone https://github.com/Geekunknown29/panorama-stitch.git
    cd panorama-stitch
    ```

3. Create and activate a virtual environment.

    **Windows PowerShell:**
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

    **Linux/macOS:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4. Install the project dependencies:

    ```bash
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

5. Verify the installation:

    ```bash
    python main.py --help
    ```

6. Run a sample three-image panorama:

    ```bash
    python main.py --input data/sample/scene1_01.jpg data/sample/scene1_02.jpg data/sample/scene1_03.jpg --output outputs/panorama.jpg
    ```

The application creates the output directory when needed. Existing panorama, metrics, and debug files are preserved; a later run receives a numbered filename such as `panorama_1.jpg` instead of overwriting the earlier result.

---

## 2. Problem Statement
Manual image alignment for panoramic stitching is error-prone, labor-intensive, and sensitive to perspective distortion and illumination shifts. Automated panoramic stitching requires solving fundamental vision challenges:
- Detecting invariant local features despite illumination, scale, and rotation variations.
- Matching descriptor vectors and filtering false correspondences.
- Estimating geometric projective transformations ($3 \times 3$ Homography matrix) while rejecting outliers.
- Mapping image coordinate spaces onto a unified global canvas without cropping valid regions or distorting geometry.
- Blending overlapping regions to eliminate seam boundaries.

This project delivers a robust software solution addressing these requirements.

---

## 3. Objectives
- **Modularity**: Maintain strict separation of concerns across input handling, preprocessing, feature extraction, matching, homography estimation, warping, blending, cropping, and evaluation.
- **Reproducibility**: Include pre-packaged synthetic overlapping datasets in `data/sample/` for immediate offline execution without external dependencies.
- **Robust Error Handling**: Provide informative human-readable diagnostic messages for edge cases (insufficient features, corrupt images, missing files, low match counts) without exposing unhandled tracebacks.
- **Educational & Viva Value**: Provide an explicit `--debug` mode exporting intermediate keypoints, match correspondence lines, homography matrices, and warped frame layers.

---

## 4. Key Features
- **Flexible CLI Arguments**: Accept explicit image file sequences (`--input img1 img2 img3`) or automatically scan directories (`--input-dir path/to/dir/`).
- **Configurable Feature Extractor**: Support for SIFT (primary, 128D floating point) and ORB (binary, 32-byte) feature algorithms via `--feature {sift,orb}`.
- **Match Filtering**: K-Nearest Neighbors ($k=2$) matching with customizable Lowe's ratio test threshold (`--ratio`).
- **RANSAC Homography Solver**: Geometric transformation estimation with RANSAC outlier rejection, determinant checks, and mean reprojection error calculations.
- **Seamless Distance-Transform Blending**: Smooth alpha-feathered blending across overlap boundaries to eliminate harsh seams.
- **Automatic Sequence Reordering**: Pairwise match graph computation (`--auto-order`) to discover optimal stitching chains automatically.
- **Quantitative Quality Metrics**: Output structured JSON report (`outputs/metrics.json`) recording keypoint counts, raw/good matches, inlier percentages, reprojection errors, and execution timings.

---

## 5. Computer Vision Concepts Used

| Concept | Description & Implementation Detail |
| :--- | :--- |
| **Image Preprocessing** | Aspect-ratio preserving spatial scaling (`max_dimension`) for working copies; BGR-to-Grayscale conversion for feature extraction. |
| **Keypoint Detection** | Locating scale-space extrema (SIFT) or FAST corner responses (ORB). |
| **Feature Descriptors** | 128-dimensional SIFT gradient orientation histograms or 256-bit binary ORB descriptors. |
| **KNN Feature Matching** | Matching descriptors across image pairs using $L_2$ Euclidean norm (SIFT) or Hamming distance (ORB). |
| **Lowe's Ratio Test** | Filtering ambiguous matches by enforcing $d(m_1) < \text{ratio} \times d(m_2)$ (default ratio: 0.75). |
| **Homography Matrix ($H$)** | $3 \times 3$ projective transformation matrix mapping points $(x, y, 1)^T \to (x', y', 1)^T$. |
| **RANSAC Outlier Rejection** | Iteratively selecting 4 random point pairs to estimate $H$ and identify inliers within reprojection threshold $\le 4.0\text{px}$. |
| **Perspective Warping** | Applying composite homography matrix $H_{composite} = H_{trans} \times H$ via `cv2.warpPerspective`. |
| **Feathered Blending** | Computing Euclidean distance-transform weight maps to produce smooth linear alpha blending over overlaps. |
| **Zero-Border Cropping** | Finding non-zero content bounding box to crop dark canvas borders. |

---

## 6. System Architecture

```
                 +-----------------------+
                 |    Input Images       |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |   Input Validation    |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |  Image Preprocessing  | (Working copy resize & Grayscale)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 | Feature Detection &   | (SIFT / ORB Keypoints & Descriptors)
                 | Descriptor Extraction |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |    Feature Matching   | (KNN k=2 & Lowe's Ratio Test)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 | Homography Estimation | (RANSAC Outlier Rejection & Matrix Validation)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |  Perspective Warping  | (Global Canvas Calculation & Homography Warp)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |    Feathered Blend    | (Distance-Transform Alpha Blending)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |   Cropping & Cleanup  | (Outer Canvas Border Removal)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |  Final Panorama &     | (outputs/panorama.jpg + metrics.json)
                 | Quantitative Metrics  |
                 +-----------------------+
```

---

## 7. Project Structure

```
panorama-builder/
├── README.md                  # Comprehensive evaluator guide & user manual
├── statement.md               # Academic problem statement, scope, and expected outcomes
├── requirements.txt           # Minimal Python dependencies (opencv-python, numpy, pytest)
├── .gitignore                 # Output and cache ignore rules
├── LICENSE                    # MIT License
├── generate_sample_data.py    # Generator script for reproducible sample datasets
├── main.py                    # CLI entrypoint & PanoramaStitcher controller
│
├── src/
│   ├── __init__.py
│   ├── input_handler.py       # Input validation & error management
│   ├── preprocessing.py       # Image resizing & grayscale conversion
│   ├── feature_detector.py    # SIFT and ORB feature detectors
│   ├── feature_matcher.py     # KNN descriptor matcher & Lowe's ratio test filter
│   ├── homography.py          # RANSAC homography estimation & matrix validation
│   ├── warper.py              # Global canvas computation & perspective warper
│   ├── blender.py             # Distance-transform feathered alpha blender
│   ├── cropper.py             # Non-zero ROI border cropper
│   ├── ordering.py            # Automatic feature-overlap sequence ordering
│   ├── evaluator.py           # Quantitative metrics compiler
│   └── utils.py               # Debug visualizer writer & console output formatter
│
├── tests/
│   ├── __init__.py
│   ├── test_input.py          # Input validation tests
│   ├── test_preprocessing.py  # Preprocessing & resizing tests
│   ├── test_features.py       # SIFT & ORB feature detector tests
│   ├── test_matching.py       # Descriptor matching & ratio filter tests
│   ├── test_homography.py     # Homography solver & RANSAC unit tests
│   ├── test_warper_blender.py # Warper canvas & blender tests
│   └── test_pipeline.py       # End-to-end integration tests
│
├── data/
│   ├── sample/                # Overlapping sample images for reproducibility
│   ├── failure_cases/         # Unrelated images for testing error handling
│   └── README.md              # Dataset origin documentation
│
├── outputs/                   # Generated panorama.jpg & metrics.json
│   └── debug/                 # Exported keypoints, matches, homography text, and warped frames
│
└── docs/                      # Architectural diagrams & academic report support
    ├── architecture.md
    ├── workflow.md
    ├── use_case.md
    ├── sequence.md
    ├── class_diagram.md
    └── project_report.md
```

---

## 8. Requirements

- **Python**: `3.10` or higher
- **Core Dependencies**:
  - `opencv-python >= 4.8.0`
  - `numpy >= 1.24.0`
  - `pytest >= 7.0.0`
- **Hardware**: CPU only (No GPU, CUDA, Docker, or external web service required).

---

## 9. Installation & Setup

1. **Clone Repository**:
   ```bash
    git clone https://github.com/Geekunknown29/panorama-stitch.git
    cd panorama-stitch
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
   ```

No API keys, databases, external services, or environment files are required. The included sample images can be used offline after dependency installation.

---

## 10. Environment Setup Verification

Verify CLI availability:
```bash
python main.py --help
```

Generate or refresh sample datasets:
```bash
python generate_sample_data.py
```

---

## 11. Running the Application

### 11.1 Two-Image Panorama
```bash
python main.py --input data/sample/scene1_01.jpg data/sample/scene1_02.jpg --output outputs/panorama.jpg
```

### 11.2 Three-Image Panorama with Debug Mode
```bash
python main.py --input data/sample/scene1_01.jpg data/sample/scene1_02.jpg data/sample/scene1_03.jpg --output outputs/panorama.jpg --debug
```

### 11.3 Preserve a Manually Supplied Order

Automatic ordering is enabled by default for three or more images. To stitch images exactly in the order supplied, add `--no-auto-order`:

```bash
python main.py --no-auto-order --input left.jpg center.jpg right.jpg --output outputs/manual_order.jpg
```

### 11.4 Directory Input Mode
```bash
python main.py --input-dir data/sample/ --output outputs/panorama.jpg
```

### 11.5 GUI File Picker

```bash
python main.py --gui
```

Select at least two overlapping images in the file picker. The GUI mode enables automatic ordering for multi-image selections.

### 11.6 ORB Feature Detector with Custom Ratio
```bash
python main.py --input-dir data/sample/ --output outputs/panorama_orb.jpg --feature orb --ratio 0.80
```

---

## 12. CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `-i`, `--input` | List | `None` | Explicit list of input image paths (at least 2 required). |
| `-d`, `--input-dir` | String | `None` | Path to directory containing input images. |
| `-o`, `--output` | String | `outputs/panorama.jpg` | Target output path for final panorama image. |
| `-f`, `--feature` | Choice | `sift` | Feature detector algorithm: `sift` or `orb`. |
| `-r`, `--ratio` | Float | `0.75` | Lowe's ratio test match filter threshold. |
| `-m`, `--max-dimension` | Int | `1600` | Max width/height dimension for working copy resizing. |
| `--auto-order` | Flag | `True` for 3+ images | Automatically reorder image sequence by validated match-graph overlap. |
| `--no-auto-order` | Flag | `False` | Disable automatic ordering and use the exact supplied image order. |
| `--gui` | Flag | `False` | Open a native file picker for image selection. |
| `--interactive` | Flag | `False` | Enter image paths interactively in the terminal. |
| `--debug` | Flag | `False` | Export intermediate debug images and matrix text files to `outputs/debug/`. |

---

## 13. Example Terminal Output

```
========================================
INTELLIGENT PANORAMA BUILDER
========================================
Images processed: 3
Feature detector: SIFT

Total keypoints:
  scene1_01.jpg: 4002
  scene1_02.jpg: 4000
  scene1_03.jpg: 4000

Successful image pairs: 2/2
Average inlier ratio: 69.56%
Processing time: 2.45 seconds

Panorama:
  Width: 2399
  Height: 1007

Status: SUCCESS

Output:
outputs/panorama.jpg
========================================
```

---

## 14. Output Description

Normal execution generates:
- `outputs/panorama.jpg` or the next available numbered filename: Final blended and cropped panoramic image.
- `outputs/metrics.json` or the next available numbered metrics filename: Structured quantitative performance metrics report.

Generated output files are ignored by Git. They remain on the local machine and are not uploaded as source files when the project is committed.

Sample `metrics.json`:
```json
{
    "status": "SUCCESS",
    "images_processed": 3,
    "feature_detector": "sift",
    "keypoints": {
        "scene1_01.jpg": 4002,
        "scene1_02.jpg": 4000,
        "scene1_03.jpg": 4000
    },
    "matches": {
        "raw": {
            "scene1_01.jpg -> scene1_02.jpg": 4000
        },
        "good": {
            "scene1_01.jpg -> scene1_02.jpg": 763
        }
    },
    "inliers": {
        "scene1_01.jpg -> scene1_02.jpg": 486
    },
    "average_inlier_ratio_percent": 69.56,
    "reprojection_errors": {
        "scene1_01.jpg -> scene1_02.jpg": 0.142
    },
    "panorama_dimensions": {
        "width": 2399,
        "height": 1007
    },
    "processing_time_seconds": 2.45
}
```

---

## 15. Debug Mode

When `--debug` is specified, the application exports diagnostic files to a run-specific debug directory beside the output, for example `outputs/panorama_debug/`:
- `keypoints_01.jpg`, `keypoints_02.jpg`, ...: Visualization of detected keypoints with orientation vectors.
- `matches_01_02.jpg`, ...: Match correspondence lines connecting feature pairs across images.
- `homography_01_02.txt`, ...: Plain text files containing the estimated $3 \times 3$ floating-point homography matrix.
- `warped_base_*.jpg`, `warped_next_*.jpg`: Intermediate perspective-transformed canvas layers.
- `panorama_before_crop.jpg`: Uncropped composite canvas showing outer black padding.

---

## 16. Testing Instructions

Run full automated unit and integration test suite using `pytest`:

```bash
python -m pytest tests/ -v
```

Test results include:
- `test_input.py`: Missing files, invalid formats, single image error handling.
- `test_preprocessing.py`: Aspect-ratio preserving resizing & grayscale conversion.
- `test_features.py`: SIFT & ORB feature extraction.
- `test_matching.py`: KNN matching and ratio test filtering.
- `test_homography.py`: Synthetic RANSAC homography recovery and degenerate cases.
- `test_warper_blender.py`: Canvas size computation & distance-transform alpha blending.
- `test_pipeline.py`: End-to-end 2-image, 3-image, resolution mismatch, and failure case tests.

---

## 17. Viva / Evaluator Q&A Reference

### Q1: Why convert images to grayscale for feature detection?
Feature detection algorithms (SIFT/ORB) identify keypoints based on local spatial intensity gradients and corner structures. Grayscale images preserve essential spatial gradient info while reducing memory consumption and computational complexity by $3\times$.

### Q2: Why is Lowe's Ratio Test necessary?
Simple nearest-neighbor matching often matches repetitive patterns (e.g., windows or bricks) to incorrect points. Lowe's ratio test checks if the closest match is significantly closer than the second-closest match ($d_1 < 0.75 \times d_2$). If $d_1 \approx d_2$, the match is ambiguous and discarded.

### Q3: What does a $3 \times 3$ Homography Matrix represent?
A Homography matrix $H$ represents a projective transformation mapping 2D homogeneous coordinates from one plane to another ($p' = H \cdot p$). It accounts for translation, rotation, scaling, affine skew, and perspective projection.

### Q4: Why is RANSAC necessary for Homography estimation?
Even after Lowe's ratio test, false matches (outliers) remain. Standard least-squares estimation would be heavily skewed by outliers. RANSAC randomly samples 4 point pairs repeatedly to find the consensus transformation model that maximizes the count of geometrically consistent inliers.

---

## 18. Limitations
- **Pure Planar Homography Assumption**: Assumes either camera rotation about its center of optical center or a purely planar physical scene. Large parallax shifts from translation in 3D non-planar scenes can cause ghosting.
- **Exposure Differences**: High intensity variance across images can result in slight tone boundaries; advanced multi-band Laplacian pyramid blending can further enhance extreme exposure differences.

---

## 19. Future Enhancements
- Cylindrical and Spherical projection warping for 360-degree panoramic capture.
- Multi-band Laplacian pyramid blending for extreme exposure transitions.
- Gain compensation and global bundle adjustment across multi-image graphs.

---

## 20. References
1. Lowe, D. G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints*. International Journal of Computer Vision (IJCV), 60(2), 91–110.
2. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). *ORB: An efficient alternative to SIFT or SURF*. IEEE International Conference on Computer Vision (ICCV).
3. Fischler, M. A., & Bolles, R. C. (1981). *Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*. Communications of the ACM, 24(6), 381–395.
4. Szeliski, R. (2010). *Computer Vision: Algorithms and Applications*. Springer Science & Business Media.
