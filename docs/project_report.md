# ACADEMIC PROJECT REPORT

## INTELLIGENT PANORAMA BUILDER
**Course**: Advanced Computer Vision  
**System Architecture**: Production-Quality Modular CLI  
**Language & Environment**: Python 3.11 / OpenCV 4.x / NumPy  

---

## 1. Cover Page
- **Project Title**: Intelligent Panorama Builder
- **Domain**: Computer Vision, Feature Detection, Image Geometry, Homography Estimation, Image Blending
- **Author**: Academic Computer Vision Project Team
- **Repository**: https://github.com/Geekunknown29/panorama-stitch
- **Evaluation Target**: Command-Line Image Stitching System

---

## 2. Introduction
Image stitching is a fundamental application in computer vision that combines multiple images with overlapping fields of view to generate a single high-resolution panoramic composite. Panoramic stitching is widely utilized in satellite imagery, digital photography, robotic vision, medical imaging, and virtual reality environment capture.

This report presents the design, mathematical formulation, implementation, and empirical evaluation of the **Intelligent Panorama Builder** system. The application enforces a complete modular pipeline: image validation, aspect-preserving spatial preprocessing, feature detection and descriptor extraction (SIFT/ORB), K-Nearest Neighbors descriptor matching with Lowe's ratio test filtering, RANSAC homography matrix estimation, global canvas perspective warping, distance-transform feathered blending, automatic border cropping, and quantitative performance evaluation.

---

## 3. Problem Statement
Capturing wide-angle physical scenes using conventional camera sensors is limited by the optical field of view. Constructing panoramic views by manually aligning images introduces parallax errors, perspective distortions, and visible exposure seams.

Automating image alignment requires solving core computer vision challenges:
1. Detecting invariant keypoints despite rotation, scale, and lighting changes.
2. Filtering out false matches caused by repetitive visual textures.
3. Estimating a $3 \times 3$ Homography matrix transformation while rejecting geometric outliers.
4. Aligning images on a unified composite coordinate canvas without distorting valid image regions.
5. Blending overlapping boundaries to eliminate seam artifacts.

---

## 4. Functional Requirements
- **FR1 Input Validation**: Accept list of file paths (`--input`) or directory (`--input-dir`); validate formats (`.jpg`, `.png`, `.bmp`, `.webp`, `.tiff`), readability, and minimum image count ($\ge 2$).
- **FR2 Image Preprocessing**: Maintain original inputs untouched; create working copies scaled to `max_dimension=1600` while preserving exact aspect ratios; convert to single-channel grayscale.
- **FR3 Feature Detection & Extraction**: Support SIFT (primary, 128D) and ORB (binary, 32-byte) detectors.
- **FR4 Feature Matching & Filtering**: KNN matching ($k=2$) with Lowe's ratio test filtering ($m.distance < ratio \times n.distance$).
- **FR5 Homography Estimation & RANSAC**: Estimate 3x3 homography matrix $H$; reject outliers; calculate inlier ratio % and mean reprojection error.
- **FR6 Perspective Warping**: Compute global bounding box dimensions, construct translation matrix $H_{trans}$, and warp images onto unified canvas.
- **FR7 Image Blending**: Create distance-transform weight maps to produce smooth linear alpha blending across overlapping boundaries.
- **FR8 Output & Metrics Generation**: Save a cropped panorama and structured metrics, allocating numbered filenames when an earlier result already exists.
- **FR9 Debug Visualization**: Export intermediate keypoint overlays, match correspondence lines, homography matrix text files, and warped frames when `--debug` is enabled.

---

## 5. Non-Functional Requirements
- **NFR1 Reliability**: Gracefully handle missing files, unsupported formats, corrupt data, low match counts, and homography failures with controlled human-readable error messages and exit code 1.
- **NFR2 Maintainability**: Modular architecture adhering to PEP 8, type hinting, class encapsulation, and zero global mutable state.
- **NFR3 Performance**: Keep working-image dimensions, canvas limits, and accumulation buffers bounded so ordinary three-image workloads remain practical on CPU hardware.
- **NFR4 Reproducibility**: Self-contained sample datasets (`data/sample/`) allowing offline execution out-of-the-box.
- **NFR5 Portability**: Machine-independent relative file paths compatible across Windows, Linux, and macOS.

---

## 6. System Architecture

```
                 +-----------------------+
                 |    Input Images       |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |   Input Validation    | (InputHandler)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |  Image Preprocessing  | (ImagePreprocessor)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 | Feature Detection &   | (FeatureDetector - SIFT/ORB)
                 | Descriptor Extraction |
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |    Feature Matching   | (FeatureMatcher - KNN & Lowe Ratio)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 | Homography Estimation | (HomographyEstimator - RANSAC)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |  Perspective Warping  | (ImageWarper & Canvas Bounds)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |    Global Warp & Blend | (ImageWarper + ImageBlender)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |   Cropping & Cleanup  | (PanoramaCropper - ROI Crop)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |  Outputs & Metrics    | (output image + metrics JSON)
                 +-----------------------+
```

---

## 7. Use Case Diagram
See [`use_case.md`](use_case.md) for the full use case diagram detailing interactions between the user, evaluator, pytest automation, and system modules.

---

## 8. Workflow Diagram
See [`workflow.md`](workflow.md) for the complete flowchart of system execution paths and error recovery branches.

---

## 9. Sequence Diagram
See [`sequence.md`](sequence.md) for the object call sequence across modules during pipeline execution.

---

## 10. Class / Component Diagram
See [`class_diagram.md`](class_diagram.md) for the complete object-oriented class structure and design abstractions.

---

## 11. Design Decisions and Rationale
1. **Classical Vision vs Deep Learning**: Selected classical SIFT/ORB + RANSAC homography over deep neural networks. Classical methods are deterministic, computationally lightweight (no GPU required), fully explainable for course evaluation, and highly accurate for planar/panoramic scenes.
2. **Validated Automatic Ordering**: For three or more images, `ImageOrderer` builds a pairwise graph from validated homographies. Edge scores combine quality and inlier count, disconnected images can be excluded, and the selected pair estimates are reused during stitching so ordering and composition do not make independent RANSAC decisions.
3. **Central Reference Composition**: Multi-image stitching maps every image into the middle image's coordinate frame. Left-side transforms are composed forward; right-side transforms use inverses. A focused unit test verifies both directions.
4. **Distance-Transform Feathered Blending**: Multi-image blending accumulates warped colors and distance weights into bounded global buffers, smoothly transitioning intensity across overlaps without repeatedly creating panorama-sized layer stacks.
5. **Defensive Output and Failure Handling**: Homography, canvas, and memory checks reject unsafe geometry before allocation. Existing output and metric files receive numbered names instead of being overwritten.

---

## 12. Implementation Details

### 12.1 Entry Point and Input Modes

`main.py` defines `PanoramaStitcher`, the application controller, and the CLI entry point. It accepts explicit files with `--input`, directory input with `--input-dir`, a native picker with `--gui`, or terminal prompts with `--interactive`. Automatic ordering is enabled by default for three or more images and can be disabled with `--no-auto-order`.

`InputHandler.validate_and_collect_paths` checks existence, file type, readability, and the minimum of two images. CLI exceptions derived from `PanoramaError` are converted into clear error messages and a non-zero exit status.

### 12.2 Preprocessing and Features

`ImagePreprocessor` reads each image and creates a working copy whose largest dimension is at most `max_dimension` (default 1600). It preserves aspect ratio, stores original and working shapes, and creates grayscale data for feature extraction. The color working copy and grayscale working copy share the same coordinate system.

`FeatureDetector` supports SIFT and ORB. SIFT is the default and is limited to approximately 4,000 features per image. `FeatureMatcher` uses OpenCV KNN matching with two neighbors and Lowe's ratio test. It returns source points, destination points, match counts, and the retained match objects.

### 12.3 Homography Estimation and Validation

`HomographyEstimator` uses `cv2.findHomography` with RANSAC to estimate a matrix mapping source coordinates to destination coordinates. It computes inlier counts, inlier ratios, and mean/median/95th-percentile/maximum reprojection errors.

Pair quality validation also checks:

- finite values, determinant sign and magnitude, and condition number;
- projected-corner convexity, area ratio, scale change, and dimensions;
- perspective terms and corner displacement relative to the destination frame;
- spatial support of the inliers.

Spatial support uses normalized bounding-box width and height, convex-hull area, and occupied quadrants. A valid edge-overlap pair may pass through sufficient width/height or hull coverage even when its inliers occupy one quadrant; a tiny concentrated cluster is still rejected.

### 12.4 Automatic Ordering and Transform Reuse

`ImageOrderer.find_optimal_order` detects features once, evaluates candidate pairs, and creates a weighted connectivity graph. Only pair estimates whose quality report is accepted become graph edges. Disconnected components are reported; with partial ordering enabled, the largest connected component is retained and excluded filenames are recorded in metrics.

Each accepted edge stores its homography, match result, quality report, and whether the estimate was computed in lower-index-to-higher-index direction. After the best chain is selected, `PanoramaStitcher` orients each cached matrix to the chain direction. Multi-image stitching reuses these validated estimates rather than re-running matching and RANSAC for the same edge. This avoids inconsistent ordering-versus-stitching decisions.

### 12.5 Multi-Image Chaining

For a chain of images, the middle image is the reference frame. If $H_{k\rightarrow k+1}$ maps image $k$ to image $k+1$, then:

$$
H_{k\rightarrow ref} = H_{k+1\rightarrow ref} H_{k\rightarrow k+1}
$$

for images to the left of the reference. For images to the right, the inverse pair transform is composed:

$$
H_{k\rightarrow ref} = H_{k-1\rightarrow ref} H_{k\rightarrow k-1}.
$$

The cumulative matrices are validated before global canvas allocation. `tests/test_homography.py` includes synthetic forward and inverse three-image checks.

### 12.6 Warping, Blending, and Cropping

`ImageWarper.calculate_multi_canvas_bounds` projects all image corners into the reference frame, computes a global bounding box, and creates a translation matrix for negative coordinates. It rejects non-positive, oversized, over-pixel-limit, or over-memory-limit canvases before allocation.

`ImageBlender.blend_multi` warps each image and a distance-transform weight map into the global canvas. It accumulates weighted color and weight buffers, then normalizes valid pixels into the final BGR panorama. `PanoramaCropper` removes empty black borders from the completed composite.

The two-image path remains separate for backward compatibility: it calculates pair bounds, warps both images, and uses the two-layer feathered blender.

### 12.7 Metrics, Debugging, and Output Files

`Evaluator` records image counts, keypoints, raw/good matches, inliers, inlier ratios, reprojection errors, dimensions, excluded images, pair-quality reports, status, and elapsed time as JSON. `DebugWriter` optionally saves keypoint drawings, match visualizations, homographies, projected corners, warped layers, and pair reports.

`PanoramaStitcher._allocate_available_filepath` prevents overwriting an existing panorama or metrics file by selecting numbered siblings such as `panorama_1.jpg`. Debug artifacts are written to a run-specific directory beside the selected output.

---

## 13. Empirical Results & Performance Evaluation

The repository includes six overlapping sample images in `data/sample/` and two unrelated failure-case images in `data/failure_cases/`. A representative demonstration is:

```bash
python main.py --input data/sample/scene1_01.jpg data/sample/scene1_02.jpg data/sample/scene1_03.jpg --output outputs/panorama.jpg
```

The generated metrics JSON records the actual detector, image dimensions, keypoint counts, raw and filtered matches, RANSAC inliers, reprojection errors, excluded images, pair-quality reports, panorama dimensions, status, and processing time. Exact values depend on the OpenCV build, CPU, input files, and detector settings; the report therefore treats the JSON file produced by each run as the authoritative measurement rather than hard-coding one historical run.

The validation workflow used during submission preparation included:

- 12 focused homography tests, including spatial edge-overlap acceptance and synthetic forward/inverse chaining;
- selected 2-image and 3-image sample pipeline tests;
- canvas safety, unrelated-image rejection, and different-resolution checks;
- CLI help verification.

Large 4-, 6-, and 15-image stress runs are intentionally not required for the basic evaluator setup. Images with insufficient overlap, strong parallax, or incompatible viewpoints should be rejected instead of being forced into a malformed panorama.

---

## 14. Testing Approach
Automated testing is implemented using `pytest` in `tests/`:
- **Unit Tests**: Verified isolated functionality for input handling, preprocessing, feature extraction, descriptor matching, RANSAC homography, canvas warping, and distance blending.
- **Integration Tests**: Cover end-to-end 2-image and 3-image sample stitching, resolution mismatch, canvas safety, and unrelated-image failure handling. Additional 4- and 6-image tests are included for broader local validation.
- **Focused validation**: The current submission validation passed the 12 homography tests and the selected lightweight pipeline checks. Run `python -m pytest tests/ -v` to execute the complete local suite when time and hardware permit.

---

## 15. Challenges Faced & Solutions
1. **Negative Coordinate Canvas Offset**: Warping images with homography often maps coordinates to negative canvas indices $(x < 0, y < 0)$.
   - *Solution*: Designed `calculate_canvas_bounds` in `warper.py` to map image corners, compute minimum negative bounds $(x_{min}, y_{min})$, and synthesize a translation matrix $H_{trans} = \begin{bmatrix} 1 & 0 & -x_{min} \\ 0 & 1 & -y_{min} \\ 0 & 0 & 1 \end{bmatrix}$ applied to composite homography.
2. **Seam Artifacts in Overlaps**: Simple image overwrite created sharp intensity seams due to minor vignetting.
   - *Solution*: Developed distance-transform alpha blending in `blender.py` where pixel weights are proportional to Euclidean distance from image borders.

---

## 16. Key Learnings & Takeaways
- **Geometric Invariance**: SIFT descriptors remain remarkably stable under affine and scale variations, enabling precise feature matching across panning camera angles.
- **Outlier Robustness**: RANSAC is indispensable for filtering out false correspondences caused by repetitive texture patterns.
- **Quantitative Evaluation**: Tracking metrics (inlier ratio %, reprojection error in pixels) provides objective proof of stitching quality without relying on subjective visual inspection alone.

---

## 17. Future Enhancements
- Cylindrical and Spherical coordinate warping for 360-degree environmental mapping.
- Laplacian pyramid multi-band blending for extreme exposure variation.
- Global bundle adjustment to minimize cumulative drift across large image graphs ($\ge 10$ images).

---

## 18. References
1. Lowe, D. G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints*. International Journal of Computer Vision (IJCV), 60(2), 91–110.
2. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). *ORB: An efficient alternative to SIFT or SURF*. IEEE International Conference on Computer Vision (ICCV).
3. Fischler, M. A., & Bolles, R. C. (1981). *Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*. Communications of the ACM, 24(6), 381–395.
4. Szeliski, R. (2010). *Computer Vision: Algorithms and Applications*. Springer Science & Business Media.
