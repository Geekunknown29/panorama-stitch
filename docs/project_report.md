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
- **Repository**: Seeding Academic Public Standard Codebase
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
- **FR8 Output & Metrics Generation**: Save cropped panorama to `outputs/panorama.jpg` and export metrics to `outputs/metrics.json`.
- **FR9 Debug Visualization**: Export intermediate keypoint overlays, match correspondence lines, homography matrix text files, and warped frames when `--debug` is enabled.

---

## 5. Non-Functional Requirements
- **NFR1 Reliability**: Gracefully handle missing files, unsupported formats, corrupt data, low match counts, and homography failures with controlled human-readable error messages and exit code 1.
- **NFR2 Maintainability**: Modular architecture adhering to PEP 8, type hinting, class encapsulation, and zero global mutable state.
- **NFR3 Performance**: Process 3 multi-megapixel images in under 5.0 seconds on standard CPU hardware.
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
                 |    Feathered Blend    | (ImageBlender - Distance Transform)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |   Cropping & Cleanup  | (PanoramaCropper - ROI Crop)
                 +-----------+-----------+
                             |
                             v
                 +-----------------------+
                 |  Outputs & Metrics    | (outputs/panorama.jpg + metrics.json)
                 +-----------------------+
```

---

## 7. Use Case Diagram
See [`docs/use_case.md`](file:///d:/Projects/panorama_stitch/docs/use_case.md) for full use case diagram detailing interactions between User, Evaluator, Pytest Automation, and system modules.

---

## 8. Workflow Diagram
See [`docs/workflow.md`](file:///d:/Projects/panorama_stitch/docs/workflow.md) for complete flowchart of system execution paths and error recovery branches.

---

## 9. Sequence Diagram
See [`docs/sequence.md`](file:///d:/Projects/panorama_stitch/docs/sequence.md) for object call sequence across modules during pipeline execution.

---

## 10. Class / Component Diagram
See [`docs/class_diagram.md`](file:///d:/Projects/panorama_stitch/docs/class_diagram.md) for complete object-oriented class structure and design abstractions.

---

## 11. Design Decisions and Rationale
1. **Classical Vision vs Deep Learning**: Selected classical SIFT/ORB + RANSAC homography over deep neural networks. Classical methods are deterministic, computationally lightweight (no GPU required), fully explainable for course evaluation, and highly accurate for planar/panoramic scenes.
2. **Sequential Pairwise Stitching**: Implemented an iterative baseline ($I_{base} \leftarrow \text{Blend}(I_{base}, I_{next})$) which ensures predictable memory consumption and simple debug artifact tracing.
3. **Distance-Transform Feathered Blending**: Replaced naive alpha overlay or hard seam cuts with distance-transform weighting map $D(x, y)$. This smoothly transitions intensity across overlapping boundaries while preserving sharp detail in central regions.

---

## 12. Implementation Details
The system is implemented across Python source modules:
- `src/input_handler.py`: File system verification and format checking.
- `src/preprocessing.py`: Spatial aspect-preserving scaling and grayscale transformation.
- `src/feature_detector.py`: SIFT/ORB keypoint and descriptor extraction.
- `src/feature_matcher.py`: KNN matching and ratio test filtering.
- `src/homography.py`: OpenCV RANSAC homography estimation and reprojection error computation.
- `src/warper.py`: Global bounding box computation and translation matrix synthesis.
- `src/blender.py`: Distance-transform weight mask generation and multi-channel alpha blending.
- `src/cropper.py`: Contour-based non-zero bounding box cropping.
- `src/evaluator.py`: Structured JSON metrics compiler.
- `src/utils.py`: Debug image renderer and formatted console reporter.

---

## 13. Empirical Results & Performance Evaluation

Execution on 3-image sample dataset (`scene1_01.jpg`, `scene1_02.jpg`, `scene1_03.jpg`):

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
            "scene1_01.jpg -> scene1_02.jpg": 4000,
            "panorama_layer_2 -> scene1_03.jpg": 4000
        },
        "good": {
            "scene1_01.jpg -> scene1_02.jpg": 763,
            "panorama_layer_2 -> scene1_03.jpg": 781
        }
    },
    "inliers": {
        "scene1_01.jpg -> scene1_02.jpg": 486,
        "panorama_layer_2 -> scene1_03.jpg": 589
    },
    "average_inlier_ratio_percent": 69.56,
    "reprojection_errors": {
        "scene1_01.jpg -> scene1_02.jpg": 0.142,
        "panorama_layer_2 -> scene1_03.jpg": 0.132
    },
    "panorama_dimensions": {
        "width": 2399,
        "height": 1007
    },
    "processing_time_seconds": 2.45
}
```

### Analysis of Results
- **Keypoint Detection**: ~4,000 SIFT keypoints detected per image view.
- **Match Filtering**: Lowe's ratio test filtered ~4,000 raw KNN matches down to ~770 high-confidence candidate matches per image pair.
- **RANSAC Inliers**: RANSAC identified ~530 geometric inliers per pair, achieving an average inlier ratio of **69.56%**.
- **Geometric Precision**: Mean reprojection errors of **0.142px** and **0.132px** demonstrate sub-pixel alignment accuracy.
- **Execution Efficiency**: Total pipeline processing time for 3 images was **2.45 seconds**.

---

## 14. Testing Approach
Automated testing is implemented using `pytest` in `tests/`:
- **Unit Tests**: Verified isolated functionality for input handling, preprocessing, feature extraction, descriptor matching, RANSAC homography, canvas warping, and distance blending.
- **Integration Tests**: Verified end-to-end 2-image, 3-image, resolution mismatch, and failure cases.
- **Results**: 100% test pass rate (19 / 19 tests passed in 10.40 seconds).

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
