# Intelligent Panorama Builder

> **Academic Computer Vision Project**

> A production-quality, modular and reproducible command-line application for image stitching using classical computer vision algorithms such as SIFT and ORB KNN matching with Lowes ratio test, RANSAC homography estimation, perspective warping, distance-transform feathered blending and quantitative evaluation.

## 1. Project Overview

The Intelligent Panorama Builder is a stand‑alone Python command-line tool created for use in Computer Vision courses. It takes overlapping images or a folder of images and automatically produces a high‑resolution, seamless panoramic image together with structured quantitative data saved in `metrics.json`.

The Intelligent Panorama Builder shows each step of the computer vision pipeline in code. It does not hide logic behind black‑box wrappers.

## How to Use

After installing the dependencies, run the program from the project folder:

### Stitch specific images

```bash
python main.py --input image_left.jpg image_center.jpg image_right.jpg --output outputs/panorama.jpg
```

For three or more images, automatic ordering is enabled by default. To preserve the exact order supplied on the command line, add `--no-auto-order`:

```bash
python main.py --no-auto-order --input image_left.jpg image_center.jpg image_right.jpg --output outputs/manual_panorama.jpg
```

### Stitch all images in a folder

```bash
python main.py --input-dir data/sample --output outputs/folder_panorama.jpg
```

### Use the graphical file picker

```bash
python main.py --gui
```

Select at least two overlapping images. The GUI enables automatic ordering for multi-image selections.

### Choose the feature detector

SIFT is the default. ORB can be selected for faster binary feature matching:

```bash
python main.py --input-dir data/sample --feature orb --ratio 0.80 --output outputs/orb_panorama.jpg
```

Use `--debug` with any command to save keypoints, matches, homographies, projected corners, and warped images. Results are written to `outputs/`; existing output files are preserved with numbered names.

## Quick Start

These steps are enough for a new user to install the Intelligent Panorama Builder without errors.

1. Install Python version three point one zero or newer. Ensure that the command `python` is available in your terminal.

2. Clone the Intelligent Panorama Builder repository and change into its directory:

```bash

git clone https://github.com/Geekunknown29/panorama-stitch.git

cd panorama-stitch

```

3.. Activate a virtual environment for the Intelligent Panorama Builder.

**Windows PowerShell:**

```powershell

python -m venv.venv

.\.venv\Scripts\Activate.ps1

```

**Linux/macOS:**

```bash

python3 -m venv.venv

source.venv/bin/activate

```

4. Install the Intelligent Panorama Builder dependencies:

```bash

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

```

5. Verify that the Intelligent Panorama Builder has been installed correctly:

```bash

python main.py --help

```

6. Run the Intelligent Panorama Builder on a sample set of three images:

```bash

python main.py --input data/sample/scene1_01.jpg data/sample/scene1_02.jpg data/sample/scene1_03.jpg --output outputs/panorama.jpg

```

The Intelligent Panorama Builder creates the output folder automatically if it does not exist. It keeps any existing panorama images, metrics and debug files. On a run it will add a number to the filename, such as `panorama_1.jpg` so it does not overwrite the previous result.

## 2. Problem Statement

Manual image alignment for stitching is error‑prone, labor‑intensive and easily affected by perspective distortion and changes in light. Automated panoramic stitching must solve key vision problems:

- Detecting local features that stay the same even when light, size or rotation change.

- Matching descriptor vectors and removing incorrect matches.

- Estimating a $3 \times 3$ Homography matrix that maps one image onto another while rejecting matches.

- Mapping the coordinate spaces of images onto one global canvas without cropping useful parts or distorting shapes.

- Blending overlapping parts so that seam lines disappear.

The Intelligent Panorama Builder delivers a software solution that meets all these requirements.

## 3. Objectives

- Modularity: Keep boundaries between input handling, preprocessing, feature extraction, matching, homography estimation, warping, blending, cropping and evaluation.

- Reproducibility: Provide ready‑made synthetic overlapping datasets in `data/sample/` so users can run the Intelligent Panorama Builder offline without needing external resources.

- Robust Error Handling: Show clear human‑readable error messages for edge cases such as too few features, damaged images missing files or low match counts and avoid raw tracebacks.

- Educational & Viva Value: Offer a `--debug` mode that exports intermediate keypoints, match lines, homography matrices and warped frame layers, for study.

## 4. Key Features

- Flexible CLI Arguments: Take a list of image files with `--input img1 img2 img3` or scan a folder automatically using `--input-dir path/to/dir/`.

- Configurable Feature Extractor: Use either SIFT, which gives 128‑dimensional floating‑point descriptors or ORB which gives 32‑byte binary descriptors, selected with `--feature {orb}`.

- Match Filtering: Perform K‑Nearest Neighbors matching with $k=2$ and adjust Lowes ratio test using the `--ratio` option.

- RANSAC Homography Solver: Compute the geometric transformation using RANSAC reject outliers check determinants and calculate the average reprojection error.

- **Seamless Distance-Transform Blending**: alpha-feathered blending across overlap boundaries to eliminate harsh seams.

- **Automatic Sequence Reordering**: Pairwise match graph computation (`--auto-order`) automatically discovers the stitching chains.

- **Quantitative Quality Metrics**: Output structured JSON report (`outputs/metrics.json`) that records keypoint counts, raw and good matches, inlier percentages, reprojection errors and execution times.

---

## 5. Computer Vision Concepts Used

| Concept Description & Implementation Detail |

| :--- | :--- |

| **Image Preprocessing** | Scale images while keeping aspect ratio (`max_dimension`) for working copies; convert from BGR to Grayscale for feature extraction. |

| **Keypoint Detection** | Locate scale‑space extrema with SIFT or FAST corner responses with ORB. |

| **Feature Descriptors** | 128‑dimensional SIFT gradient orientation. 256‑Bit binary ORB descriptors. |

*KNN Feature Matching** | Match descriptors across image pairs using L2 Euclidean norm (SIFT) or Hamming distance (ORB). |

*Lowes Ratio Test** | Filter ambiguous matches by enforcing \(d(m_1) < \text{ratio} \times d(m_2)\) (default ratio: 0.75). |

*Homography Matrix (H)** | \(3 \times 3\) projective transformation matrix mapping points \((x, y, 1)^T \to (x' y' 1)^T\). |

**RANSAC Outlier Rejection** | Iteratively select 4 random point pairs to estimate \(H\) and identify inliers within reprojection threshold \(\le 4.0\text{px}\). |

*Perspective Warping** | Apply composite homography matrix \(H_{\text{composite}} = H_{\text{trans}} \times H\) via `cv2.warpPerspective`.

| **Feathered Blending** | Compute Euclidean distance‑transform weight maps to produce smooth linear alpha blending over overlaps.

| **Zero‑Border Cropping** | Find non‑zero content bounding box to crop dark canvas borders. |

---

## 6. System Architecture

```

+-----------------------+

Input Images |

+-----------+-----------+

|

v

+-----------------------+

| Input Validation |

+-----------+-----------+

|

v

+-----------------------+

| Image Preprocessing | (Working copy resize & Grayscale)

+-----------+-----------+

v

+-----------------------+

| Feature Detection & | (SIFT / ORB Keypoints & Descriptors)

| Descriptor Extraction |

+-----------+-----------+

|

v

+-----------------------+

| Feature Matching | (KNN k=2 & Lowes Ratio Test)

+-----------+-----------+

v

+-----------------------+

| Homography Estimation | (RANSAC Outlier Rejection & Matrix Validation)

+-----------+-----------+

v

+-----------------------+

| Perspective Warping | (Global Canvas Calculation & Homography Warp)

+-----------+-----------+

v

+-----------------------+

| Feathered Blend | (Distance‑Transform Alpha Blending)

+-----------+-----------+

v

+-----------------------+

| Cropping & Cleanup | (Outer Canvas Border Removal)

+-----------+-----------+

v

+-----------------------+

| Final Panorama & | (outputs/panorama.jpg + metrics.json)

| Quantitative Metrics

+-----------------------+

```

---

## 7. Project Structure

```

panorama-builder/

├── README.md # Comprehensive evaluator guide & user manual

├── statement.md # Academic problem statement, scope and expected outcomes

├── requirements.txt # Minimal Python dependencies (opencv‑python, numpy, pytest)

├──.gitignore # Output and cache ignore rules

├── LICENSE # MIT License

├── generate_sample_data.py # Generator script, for sample datasets

├── main.py # CLI entrypoint & PanoramaStitcher controller

│

├── src/

│ ├── __init__.py

│ ├── input_handler.py # Input validation & error management

│ ├── preprocessing.py # Image resizing & grayscale conversion

│ ├── feature_detector.py # SIFT and ORB feature detectors

```

│ ├── feature_matcher.py # KNN descriptor matcher & Lowes ratio test filter

│ ├── homography.py # RANSAC homography estimation & matrix validation

│ ├── warper.py # canvas computation & perspective warper

│ ├── blender.py # Distance-transform feathered alpha blender

│ ├── cropper.py # Non-zero ROI border cropper

│ ├── ordering.py # Automatic feature-overlap sequence ordering

│ ├── evaluator.py # Quantitative metrics compiler

│ └── utils.py # Debug visualizer writer & console output formatter

│

├── tests/

│ ├── __init__.py

│ ├── test_input.py # Input validation tests

│ ├── test_preprocessing.py # Preprocessing & resizing tests

│ ├── test_features.py # SIFT & ORB feature detector tests

│ ├── test_matching.py # Descriptor matching & ratio filter tests

│ ├── test_homography.py # Homography solver & RANSAC unit tests

│ ├── test_warper_blender.py # Warper canvas & blender tests

│ └─ test_pipeline.py # End‑to‑end integration tests

│

├── data/

│ ├── sample/ # Overlapping sample images for reproducibility

│ ├── failure_cases/ # Unrelated images for testing error handling

│ └── README.md # Dataset origin documentation

│

├── outputs/ # Generated panorama.jpg & metrics.json

│ └── debug/ # Exported keypoints, matches, homography text and warped frames

│

└── docs/ # Architectural diagrams & academic report support

├── architecture.md

├── workflow.md

├── use_case.md

├── sequence.md

├── class_diagram.md

└── project_report.md

```

---

## 8. Requirements

- **Python**: `3.10` or

- **Core Dependencies**:

`Opencv-python >= 4.8.0`

`Numpy >= 1.24.0`

`Pytest >= 7.0.0`

- **Hardware**: CPU only (No GPU, CUDA, Docker or external web service required).

---

## 9.. Setup

1. **Clone Repository**:

```bash

git clone https://github.com/Geekunknown29/panorama-stitch.git

cd panorama-stitch

```

2. **. Activate Virtual Environment**:

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

No API keys, databases, external services or environment files are required. The included sample images can be used offline after dependency installation.

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

### 11.1 Two‑Image Panorama

```bash

python main.py --input data/sample/scene1_01.jpg data/sample/scene1_02.jpg --output outputs/panorama.jpg

```

### 11.2 Three‑Image Panorama with Debug Mode

```bash

python main.py --input data/sample/scene1_01.jpg data/sample/scene1_02.jpg data/sample/scene1_03.jpg --output outputs/panorama.jpg --debug

```

### 11.3 Preserve a Supplied Order

Automatic ordering is enabled by default for three or more images. To stitch images in the order supplied add `--no-auto-order`:

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

Select least two overlapping images, in the file picker. The GUI mode enables ordering for multi‑image selections.

### 11.6 ORB Feature Detector with Custom Ratio

```bash

python main.py --input-dir data/sample/ --output outputs/panorama_orb.jpg --feature orb --ratio 0.80

```

---

## 12. CLI Arguments

| Argument | Type | Default | Description |

| :--- | :--- | :--- | :---

| `-i` `--input` | List | `None` Explicit list of input image paths at least two required. |

| `-D` `--input-dir` String | `None` | Path to directory containing input images. |

| `-O` `--output` String | `outputs/panorama.jpg` | Target output path for final panorama image.

| `-F` `--feature` | Choice | `sift` | Feature detector algorithm: `sift` or `orb`.

| `-R` `--ratio` | Float | `0.75` | Lowes ratio test match filter threshold.

| `-M` `--max-dimension` | Int 1600` | Max width/height dimension for working copy resizing. |

| `--Auto-order` | Flag True` for 3+ images | Automatically reorder image sequence based on validated match‑graph overlap. |

--No-auto-order` | Flag | `False` | Disable automatic ordering and use the exact supplied image order.

| `--Gui` | Flag | `False` | Open a native file picker, for image selection.

| `--Interactive` | Flag | `False` | Enter image paths interactively in the terminal.

| `--Debug` | Flag | `False` | Export intermediate debug images and matrix text files to `outputs/debug/`. |

---

## 13. Example Terminal Output

```

========================================

INTELLIGENT PANORAMA BUILDER

========================================

Images processed: 3

Feature detector: SIFT

keypoints:

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

When the program runs normally it creates:

- `outputs/panorama.jpg` or the next available numbered filename: blended and cropped panoramic image.

- `outputs/metrics.json` or the next available numbered metrics filename: Structured quantitative performance metrics report.

Generated output files are ignored by Git staying on the local machine and not uploaded when the project is committed.

Sample `metrics.json`:

```json

{

"status": "SUCCESS"

"images_processed": 3

"feature_detector": "sift"

"keypoints": {

"scene1_01.jpg": 4002

"scene1_02.jpg": 4000

"scene1_03.jpg": 4000

}

"matches": {

"raw": {

"scene1_01.jpg -> scene1_02.jpg": 4000

}

": {

"scene1_01.jpg -> scene1_02.jpg": 763

}

}

"inliers": {

"scene1_01.jpg -> scene1_02.jpg": 486

}

"average_inlier_ratio_percent": 69.56

"reprojection_errors": {

"scene1_01.jpg -> scene1_02.jpg": 0.142

}

"panorama_dimensions": {

"width": 2399

"height": 1007

}

"processing_time_seconds": 2.45

}

```

---

## 15. Debug Mode

When the option `--debug` is chosen the software writes files to a run‑specific debug directory next to the output for example `outputs/panorama_debug/`:

- `keypoints_01.jpg` `keypoints_02.jpg` … : these images show the detected keypoints together with their orientation vectors.

- `matches_01_02.jpg` … : these images display lines that connect matched feature pairs between images.

- `homography_01_02.txt` … : plain text files that hold the estimated 3 × 3 floating‑point homography matrix.

- `warped_base_*.jpg` `warped_next_*.jpg` : intermediate layers that show the canvas after perspective transformation.

- `panorama_before_crop.jpg` : an uncropped composite canvas that displays the outer black padding.

---

## 16. Testing Instructions

Run the automated unit and integration test suite with `pytest`:

```bash

python -m pytest tests/ -v

```

Test results include:

- `test_input.py` : tests how the program handles missing files, bad formats and errors when only one image is provided.

- `test_preprocessing.py` : tests resizing that keeps the aspect ratio and converting images to grayscale.

- `test_features.py` : tests extraction of features using SIFT and ORB.

- `test_matching.py` : tests KNN matching and the ratio test used to filter matches.

- `test_homography.py` : tests recovery of homography with synthetic data using RANSAC and checks degenerate cases.

- `test_warper_blender.py` : tests how the canvas size is computed and how distance‑transform alpha blending works.

- `test_pipeline.py` : end‑to‑end tests for two‑image and three‑image panoramas, resolution mismatches and failure scenarios.

---

## 17. Viva / Evaluator Q&A Reference

### Q1: Why convert images to grayscale for feature detection?

Feature detection methods such as SIFT and ORB find keypoints by looking at intensity changes and corner shapes. Using grayscale images keeps the spatial gradient information and cuts memory usage and computation time by a factor of three.

### Q2: Why is Lowes Ratio Test necessary?

When using nearest‑neighbor matching repeated patterns like windows or bricks can be matched incorrectly. Lowes ratio test compares the distance of the match to the distance of the second‑closest match. If the first distance is than 75 percent of the second distance the match is accepted; otherwise it is considered ambiguous and removed.

### Q3: What does a 3 × 3 Homography Matrix represent?

A 3 × 3 homography matrix, written as **H** describes a transformation that maps 2‑D homogeneous coordinates from one plane to another (p′ = H · p). It covers translation, rotation, scaling affine skew and perspective projection.

### Q4: Why's RANSAC necessary for Homography estimation?

After applying Lowes ratio test some incorrect matches still exist. If we used ordinary least‑squares estimation these outliers would distort the result. RANSAC works by sampling sets of four point pairs many times looking for the transformation that has the most geometrically consistent inliers. This gives a homography estimate.

---

## 18. Limitations

- **Pure planar homography assumption** : the method assumes that the camera only rotates around its centre or that the scene is flat. When there is a parallax shift from translation in a 3‑D non‑planar scene ghosting can appear.

- **Exposure differences** : large intensity differences between images can create tone boundaries. Using an advanced multi‑band Laplacian pyramid blending could improve the handling of extreme exposure changes.

---

## 19. Future Enhancements

- Cylindrical and spherical projection warping to enable 360‑degree capture.

- Multi‑band Laplacian pyramid blending to manage extreme exposure transitions.

- Gain compensation and global bundle adjustment over many images.

---

## 20. References

1. Lowe, D. G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints*. International Journal of Computer Vision (IJCV) 60(2) 91–110.

2. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). *ORB: An alternative to SIFT or SURF*. IEEE International Conference on Computer Vision (ICCV).

3. Fischler, M. A., & Bolles, R. C. (1981). *Random Sample Consensus: A Paradigm for Model Fitting with Applications, to Image Analysis and Automated Cartography*. Communications of the ACM, 24(6) 381–395.

4. Szeliski, R. (2010). *Computer Vision: Algorithms and Applications*. Springer Science & Business Media.