"""
Intelligent Panorama Builder - Main CLI Entrypoint & Pipeline Controller.

Command-line application for stitching multi-image panoramic views using classical
computer vision techniques (SIFT/ORB, KNN matching, RANSAC Homography, Perspective Warping,
Feathered Blending, and Quantitative Metrics Evaluation).
"""

import sys
import argparse
import os
import cv2
import numpy as np

from src.input_handler import InputHandler, PanoramaError, InputValidationError
from src.preprocessing import ImagePreprocessor, ProcessedImage
from src.feature_detector import FeatureDetector, FeatureExtractionError
from src.feature_matcher import FeatureMatcher, FeatureMatchingError
from src.homography import HomographyEstimator, HomographyEstimationError
from src.warper import ImageWarper, WarpingError
from src.blender import ImageBlender, BlendingError
from src.cropper import PanoramaCropper
from src.ordering import ImageOrderer, OrderingError
from src.evaluator import Evaluator
from src.utils import DebugWriter, print_terminal_summary


class PanoramaStitcher:
    """Orchestrates the complete Computer Vision panorama stitching pipeline."""

    def __init__(
        self,
        feature_algorithm: str = "sift",
        ratio_threshold: float = 0.75,
        max_dimension: int = 1600,
        auto_order: bool = True,
        debug_mode: bool = False
    ):
        self.feature_algorithm = feature_algorithm
        self.ratio_threshold = ratio_threshold
        self.max_dimension = max_dimension
        self.auto_order = auto_order
        self.debug_mode = debug_mode

        self.preprocessor = ImagePreprocessor(max_dimension=max_dimension)
        self.detector = FeatureDetector(algorithm=feature_algorithm)
        self.matcher = FeatureMatcher(algorithm=feature_algorithm, ratio_threshold=ratio_threshold)
        self.homography_estimator = HomographyEstimator(ransac_reproj_threshold=4.0)
        self.warper = ImageWarper()
        self.blender = ImageBlender()
        self.orderer = ImageOrderer(detector=self.detector, matcher=self.matcher)
        self.evaluator = Evaluator()
        self.debug_writer = DebugWriter() if debug_mode else None

    @staticmethod
    def _allocate_available_filepath(filepath: str) -> str:
        """Return filepath or the next numbered sibling when it already exists."""
        if not os.path.exists(filepath):
            return filepath

        directory, filename = os.path.split(filepath)
        stem, extension = os.path.splitext(filename)
        index = 1
        while True:
            candidate = os.path.join(directory, f"{stem}_{index}{extension}")
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def stitch(
        self,
        input_files: list = None,
        input_dir: str = None,
        output_filepath: str = "outputs/panorama.jpg"
    ) -> str:
        """Executes end-to-end panorama stitching pipeline."""
        output_filepath = self._allocate_available_filepath(output_filepath)
        if self.debug_mode and self.debug_writer:
            output_dir = os.path.dirname(os.path.abspath(output_filepath))
            output_stem = os.path.splitext(os.path.basename(output_filepath))[0]
            self.debug_writer.debug_dir = os.path.join(output_dir, f"{output_stem}_debug")

        self.evaluator.start_timer()
        metrics = self.evaluator.metrics
        metrics.feature_detector = self.feature_algorithm

        image_paths = InputHandler.validate_and_collect_paths(
            input_files=input_files,
            input_dir=input_dir
        )
        metrics.images_count = len(image_paths)

        processed_images = []
        for path in image_paths:
            proc_img = self.preprocessor.preprocess(path)
            processed_images.append(proc_img)
            metrics.input_image_dimensions[proc_img.filename] = list(proc_img.original_shape)

        image_features = []
        for idx, proc_img in enumerate(processed_images):
            kp, des = self.detector.detect_and_compute(proc_img.gray_img)
            image_features.append((kp, des))
            metrics.keypoints_per_image[proc_img.filename] = len(kp)

            if self.debug_mode and self.debug_writer:
                kp_img = self.detector.draw_keypoints(proc_img.color_img, kp)
                self.debug_writer.save_image(
                    f"feature_keypoints/keypoints_{idx + 1:02d}_{proc_img.filename}",
                    kp_img
                )

        # Automatic ordering for N >= 3 (does NOT rely on filename order)
        if self.auto_order and len(processed_images) > 2:
            ordering_result = self.orderer.find_optimal_order(
                processed_images,
                precomputed_features=image_features,
                allow_partial=True
            )
            if ordering_result.excluded_filenames:
                metrics.excluded_images = ordering_result.excluded_filenames
                print(
                    f"\n{len(ordering_result.ordered_images)} images successfully connected. "
                    f"{len(ordering_result.excluded_filenames)} images were excluded due to "
                    f"insufficient reliable overlap.\n"
                )

            old_to_new = {id(img): idx for idx, img in enumerate(processed_images)}
            cached_pair_results = {}
            for chain_idx in range(len(ordering_result.ordered_images) - 1):
                source_img = ordering_result.ordered_images[chain_idx]
                target_img = ordering_result.ordered_images[chain_idx + 1]
                source_idx = old_to_new[id(source_img)]
                target_idx = old_to_new[id(target_img)]
                pair_key_indices = tuple(sorted((source_idx, target_idx)))
                cached = ordering_result.pairwise_results.get(pair_key_indices)
                if cached is None:
                    continue
                homo_res, match_res, source_is_lower = cached
                desired_source_is_lower = source_idx < target_idx
                H_oriented = homo_res.matrix
                if source_is_lower != desired_source_is_lower:
                    H_oriented = np.linalg.inv(H_oriented)
                cached_pair_results[chain_idx] = (homo_res, match_res, H_oriented)

            new_features = []
            for img in ordering_result.ordered_images:
                old_idx = old_to_new[id(img)]
                new_features.append(image_features[old_idx])
            processed_images = ordering_result.ordered_images
            image_features = new_features
            metrics.images_count = len(processed_images)

        N = len(processed_images)

        if N < 2:
            raise InputValidationError(
                "After ordering and exclusion, fewer than 2 images remain for stitching."
            )

        if N == 2:
            current_panorama_img = self._stitch_two_images(
                processed_images, image_features, metrics
            )
        else:
            current_panorama_img = self._stitch_multi_images(
                processed_images, image_features, metrics,
                cached_pair_results if self.auto_order else None
            )

        if self.debug_mode and self.debug_writer:
            self.debug_writer.save_image("intermediate_panorama/panorama_before_crop.jpg", current_panorama_img)

        final_panorama = PanoramaCropper.crop_black_borders(current_panorama_img)

        output_dir = os.path.dirname(os.path.abspath(output_filepath))
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_filepath, final_panorama)

        h_final, w_final = final_panorama.shape[:2]
        metrics.panorama_dimensions = [w_final, h_final]
        metrics.status = "SUCCESS"
        self.evaluator.stop_timer()

        metrics_file = self._allocate_available_filepath(os.path.join(output_dir, "metrics.json"))
        self.evaluator.save_json(metrics_file)

        if self.debug_mode and self.debug_writer:
            self.debug_writer.save_metrics_json()

        print_terminal_summary(metrics, output_filepath)
        return output_filepath

    def _record_pair_quality(self, pair_key: str, homo_res, metrics) -> None:
        if homo_res.quality_report is None:
            return
        report = homo_res.quality_report
        metrics.pair_quality_reports[pair_key] = report.to_dict()
        if self.debug_mode and self.debug_writer:
            self.debug_writer.save_pair_metrics(report.to_dict())

    def _stitch_two_images(self, processed_images, image_features, metrics):
        """
        Standard 2-image stitching pipeline. PRESERVED from original working implementation.

        Homography direction: H maps img2 coordinates → img1 coordinates.
        """
        img1_data = processed_images[0]
        img2_data = processed_images[1]
        kp1, des1 = image_features[0]
        kp2, des2 = image_features[1]
        pair_key = f"{img1_data.filename} -> {img2_data.filename}"

        print(f"\n--- Processing Image Pair: {pair_key} ---")
        print(f"  Keypoints in {img1_data.filename}: {len(kp1)}")
        print(f"  Keypoints in {img2_data.filename}: {len(kp2)}")

        match_res = self.matcher.match(
            keypoints1=kp2,
            descriptors1=des2,
            keypoints2=kp1,
            descriptors2=des1,
            img1_name=img2_data.filename,
            img2_name=img1_data.filename,
            min_good_matches=10
        )

        metrics.raw_matches_per_pair[pair_key] = match_res.raw_matches_count
        metrics.good_matches_per_pair[pair_key] = match_res.good_matches_count
        print(f"  Raw matches: {match_res.raw_matches_count}")
        print(f"  Good matches: {match_res.good_matches_count}")

        if self.debug_mode and self.debug_writer:
            match_vis = self.matcher.draw_matches(
                img2_data.color_img, kp2,
                img1_data.color_img, kp1,
                match_res.good_matches
            )
            self.debug_writer.save_image("feature_matches/matches_01_02.jpg", match_vis)

        homo_res = self.homography_estimator.estimate(
            src_pts=match_res.src_pts,
            dst_pts=match_res.dst_pts,
            pair_name=pair_key,
            min_inliers=10,
            src_shape=img2_data.color_img.shape[:2],
            dst_shape=img1_data.color_img.shape[:2],
            good_matches=match_res.good_matches_count,
            validate=True
        )

        metrics.inliers_per_pair[pair_key] = homo_res.inliers_count
        metrics.inlier_ratios_per_pair[pair_key] = round(homo_res.inlier_ratio, 2)
        metrics.reprojection_errors_per_pair[pair_key] = round(homo_res.reprojection_error, 3)

        if homo_res.quality_report:
            HomographyEstimator.print_pair_quality_report(homo_res.quality_report)

        self._record_pair_quality(pair_key, homo_res, metrics)

        proj_corners = self.homography_estimator.project_corners(img2_data.color_img.shape[:2], homo_res.matrix)
        print(f"  Projected corner coordinates:\n{proj_corners}")

        x_min, y_min, c_w, c_h, _ = self.warper.calculate_canvas_bounds(
            img1_data.color_img.shape, img2_data.color_img.shape, homo_res.matrix
        )
        est_ram = self.warper.estimate_canvas_memory_mb(c_w, c_h)
        print(f"  Estimated panorama dimensions: {c_w} x {c_h} (Estimated RAM: {est_ram:.1f} MB)\n")

        if self.debug_mode and self.debug_writer:
            self.debug_writer.save_homography(
                "homographies/homography_01_02.txt",
                homo_res.matrix,
                header_info=f"Homography for {pair_key}"
            )
            self.debug_writer.save_projected_corners(
                "projected_corners/corners_01_02.txt",
                proj_corners,
                header_info=f"Projected corners for {pair_key}"
            )

        warp_res = self.warper.warp_pair(
            img1=img1_data.color_img,
            img2=img2_data.color_img,
            H=homo_res.matrix
        )

        if self.debug_mode and self.debug_writer:
            self.debug_writer.save_image("warped_images/warped_base.jpg", warp_res.warped_img1)
            self.debug_writer.save_image("warped_images/warped_next.jpg", warp_res.warped_img2)

        blended_panorama = self.blender.blend(warp_res.warped_img1, warp_res.warped_img2)
        return blended_panorama

    def _stitch_multi_images(
        self, processed_images, image_features, metrics, cached_pair_results=None
    ):
        """
        Multi-image stitching with central reference anchor and correct homography chaining.

        HOMOGRAPHY DIRECTION CONVENTION:
        H_{k→k+1} = findHomography(src=pts_in_image_k, dst=pts_in_image_{k+1})
        p_{k+1} = H_{k→k+1} @ p_k

        CHAIN COMPOSITION TO REFERENCE (ref = N // 2):
        H_to_ref[ref] = Identity
        For k < ref: H_to_ref[k] = H_to_ref[k+1] @ H_{k→k+1}
        For k > ref: H_to_ref[k] = H_to_ref[k-1] @ inv(H_{k-1→k})
        """
        N = len(processed_images)
        ref_idx = N // 2
        ref_name = processed_images[ref_idx].filename

        print(f"\n--- Multi-Image Stitching Pipeline ({N} Images) ---")
        print(f"Reference coordinate anchor: Image {ref_idx + 1} ({ref_name})")
        print(f"Homography convention: H_{{k→k+1}} = findHomography(pts_k, pts_{{k+1}})\n")

        H_pair = [None] * (N - 1)

        for k in range(N - 1):
            src_img = processed_images[k]
            dst_img = processed_images[k + 1]
            kp_src, des_src = image_features[k]
            kp_dst, des_dst = image_features[k + 1]
            pair_key = f"{src_img.filename} → {dst_img.filename}"

            print(f"Pair [{k+1} → {k+2}]: {pair_key}")
            print(f"  Keypoints: {len(kp_src)} & {len(kp_dst)}")

            cached = cached_pair_results.get(k) if cached_pair_results else None
            if cached is not None:
                homo_res, match_res, H_pair_oriented = cached
                print("  Reusing validated ordering homography")
            else:
                match_res = self.matcher.match(
                    keypoints1=kp_src,
                    descriptors1=des_src,
                    keypoints2=kp_dst,
                    descriptors2=des_dst,
                    img1_name=src_img.filename,
                    img2_name=dst_img.filename,
                    min_good_matches=10
                )

                homo_res = self.homography_estimator.estimate(
                    src_pts=match_res.src_pts,
                    dst_pts=match_res.dst_pts,
                    pair_name=pair_key,
                    min_inliers=10,
                    src_shape=src_img.color_img.shape[:2],
                    dst_shape=dst_img.color_img.shape[:2],
                    good_matches=match_res.good_matches_count,
                    validate=True
                )
                H_pair_oriented = homo_res.matrix

            metrics.raw_matches_per_pair[pair_key] = match_res.raw_matches_count
            metrics.good_matches_per_pair[pair_key] = match_res.good_matches_count
            print(f"  Raw matches: {match_res.raw_matches_count}, "
                  f"Good matches: {match_res.good_matches_count}")

            metrics.inliers_per_pair[pair_key] = homo_res.inliers_count
            metrics.inlier_ratios_per_pair[pair_key] = round(homo_res.inlier_ratio, 2)
            metrics.reprojection_errors_per_pair[pair_key] = round(homo_res.reprojection_error, 3)

            if homo_res.quality_report:
                HomographyEstimator.print_pair_quality_report(homo_res.quality_report)

            self._record_pair_quality(pair_key, homo_res, metrics)

            H_pair[k] = H_pair_oriented

            proj_corners = self.homography_estimator.project_corners(
                src_img.color_img.shape[:2], H_pair_oriented
            )
            print(f"  Projected corners of {src_img.filename} in {dst_img.filename}'s frame:")
            print(f"    TL={proj_corners[0]}, BL={proj_corners[1]}, "
                  f"BR={proj_corners[2]}, TR={proj_corners[3]}")

            if self.debug_mode and self.debug_writer:
                match_vis = self.matcher.draw_matches(
                    src_img.color_img, kp_src,
                    dst_img.color_img, kp_dst,
                    match_res.good_matches
                )
                self.debug_writer.save_image(
                    f"feature_matches/matches_{k+1:02d}_{k+2:02d}.jpg", match_vis
                )
                self.debug_writer.save_homography(
                    f"homographies/H_{k+1:02d}_to_{k+2:02d}.txt",
                    H_pair_oriented,
                    header_info=f"H_{{k→k+1}}: {pair_key}"
                )
                self.debug_writer.save_projected_corners(
                    f"projected_corners/corners_{k+1:02d}_{k+2:02d}.txt",
                    proj_corners,
                    header_info=f"Projected corners: {pair_key}"
                )

            print()

        H_to_ref = [None] * N
        H_to_ref[ref_idx] = np.eye(3, dtype=np.float64)

        print(f"--- Chaining to reference image {ref_idx + 1} ({ref_name}) ---")

        for k in range(ref_idx - 1, -1, -1):
            H_k_to_kp1 = H_pair[k]
            H_to_ref[k] = H_to_ref[k + 1] @ H_k_to_kp1
            print(f"  H_to_ref[{k+1}] = H_to_ref[{k+2}] @ H_{{{k+1}→{k+2}}}")

            is_valid, reason = self.homography_estimator.validate_homography(
                H_to_ref[k], processed_images[k].color_img.shape[:2],
                dst_shape=processed_images[ref_idx].color_img.shape[:2]
            )
            if not is_valid:
                raise HomographyEstimationError(
                    f"Cumulative homography for image {k+1} ({processed_images[k].filename}) "
                    f"to reference is invalid: {reason}\n"
                    f"Panorama generation stopped safely.\n"
                    f"The estimated panorama dimensions are unreasonable.\n"
                    f"Possible causes:\n"
                    f"- unstable homography\n"
                    f"- insufficient overlap\n"
                    f"- incorrect image order\n"
                    f"- excessive perspective distortion."
                )

        for k in range(ref_idx + 1, N):
            H_km1_to_k = H_pair[k - 1]
            H_k_to_km1 = np.linalg.inv(H_km1_to_k)
            H_to_ref[k] = H_to_ref[k - 1] @ H_k_to_km1
            print(f"  H_to_ref[{k+1}] = H_to_ref[{k}] @ inv(H_{{{k}→{k+1}}})")

            is_valid, reason = self.homography_estimator.validate_homography(
                H_to_ref[k], processed_images[k].color_img.shape[:2],
                dst_shape=processed_images[ref_idx].color_img.shape[:2]
            )
            if not is_valid:
                raise HomographyEstimationError(
                    f"Cumulative homography for image {k+1} ({processed_images[k].filename}) "
                    f"to reference is invalid: {reason}\n"
                    f"Panorama generation stopped safely.\n"
                    f"The estimated panorama dimensions are unreasonable.\n"
                    f"Possible causes:\n"
                    f"- unstable homography\n"
                    f"- insufficient overlap\n"
                    f"- incorrect image order\n"
                    f"- excessive perspective distortion."
                )

        print()

        shapes = [p.color_img.shape[:2] for p in processed_images]
        try:
            multi_bounds = self.warper.calculate_multi_canvas_bounds(shapes, H_to_ref)
        except WarpingError as e:
            raise WarpingError(
                f"Panorama generation stopped safely.\n"
                f"The estimated panorama dimensions are unreasonable.\n"
                f"{str(e)}\n"
                f"Possible causes:\n"
                f"- unstable homography\n"
                f"- insufficient overlap\n"
                f"- incorrect image order\n"
                f"- excessive perspective distortion."
            ) from e

        print(f"--- Global Panorama Canvas ---")
        print(f"  Dimensions: {multi_bounds.canvas_width} x {multi_bounds.canvas_height}")
        print(f"  Estimated RAM: {multi_bounds.estimated_memory_mb:.1f} MB")

        for k in range(N):
            H_final_k = multi_bounds.translation_matrix @ H_to_ref[k]
            proj = self.homography_estimator.project_corners(
                processed_images[k].color_img.shape[:2], H_final_k
            )
            x_range = f"x:[{proj[:, 0].min():.0f}, {proj[:, 0].max():.0f}]"
            y_range = f"y:[{proj[:, 1].min():.0f}, {proj[:, 1].max():.0f}]"
            print(f"  Image {k+1} ({processed_images[k].filename}): {x_range}, {y_range}")

            if self.debug_mode and self.debug_writer:
                self.debug_writer.save_projected_corners(
                    f"projected_corners/global_{k+1:02d}_{processed_images[k].filename}.txt",
                    proj,
                    header_info=f"Global projected corners: {processed_images[k].filename}"
                )
        print()

        H_composites = [multi_bounds.translation_matrix @ H for H in H_to_ref]
        images_list = [p.color_img for p in processed_images]

        if self.debug_mode and self.debug_writer:
            for k in range(N):
                warped_k = cv2.warpPerspective(
                    images_list[k], H_composites[k],
                    (multi_bounds.canvas_width, multi_bounds.canvas_height),
                    flags=cv2.INTER_LINEAR
                )
                self.debug_writer.save_image(
                    f"warped_images/warped_{k+1:02d}_{processed_images[k].filename}",
                    warped_k
                )
                del warped_k

        current_panorama_img = self.blender.blend_multi(
            images=images_list,
            H_composites=H_composites,
            canvas_shape=(multi_bounds.canvas_height, multi_bounds.canvas_width)
        )

        return current_panorama_img


def build_cli_parser() -> argparse.ArgumentParser:
    """Builds clean command-line interface argument parser."""
    parser = argparse.ArgumentParser(
        prog="Intelligent Panorama Builder",
        description="Academic Computer Vision Panorama Generator using classical feature matching and homography.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/sample/scene1_01.jpg data/sample/scene1_02.jpg --output outputs/panorama.jpg
  python main.py --input-dir data/sample/ --output outputs/panorama.jpg --debug
  python main.py --gui
  python main.py --interactive
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "-i", "--input",
        nargs="+",
        metavar="IMAGE",
        help="List of paths to input image files (at least 2 required)."
    )
    input_group.add_argument(
        "-d", "--input-dir",
        metavar="DIR",
        help="Directory containing input image files."
    )
    input_group.add_argument(
        "-g", "--gui",
        action="store_true",
        help="Open native OS File Explorer window ('This PC') to pick image files."
    )
    input_group.add_argument(
        "--interactive", "-cli",
        action="store_true",
        help="Interactively prompt for image paths in terminal from any viewpoint/order."
    )

    parser.add_argument(
        "-o", "--output",
        default="outputs/panorama.jpg",
        metavar="PATH",
        help="Output filepath for generated panorama (default: outputs/panorama.jpg)."
    )
    parser.add_argument(
        "-f", "--feature",
        choices=["sift", "orb"],
        default="sift",
        help="Feature detection algorithm: 'sift' or 'orb' (default: sift)."
    )
    parser.add_argument(
        "-r", "--ratio",
        type=float,
        default=0.75,
        help="Lowe's ratio test match filter threshold (default: 0.75)."
    )
    parser.add_argument(
        "-m", "--max-dimension",
        type=int,
        default=1600,
        help="Maximum width or height dimension for working copy resizing (default: 1600)."
    )
    parser.add_argument(
        "--auto-order",
        action="store_true",
        default=None,
        help="Automatically determine optimal stitching order based on feature overlap (default for 3+ images)."
    )
    parser.add_argument(
        "--no-auto-order",
        action="store_true",
        help="Use input order as-is (not recommended for unordered multi-image sets)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to save intermediate keypoints, matches, homography text, and warped frames."
    )

    return parser


def open_file_picker() -> list:
    """Launches standard OS File Explorer GUI dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("  [!] Error: 'tkinter' standard module unavailable for graphical file picking.")
        return []

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("\n========================================")
    print("OPENING FILE EXPLORER WINDOW...")
    print("========================================")
    print("Navigate directories in the pop-up window and select images for your panorama.")
    print("Tip: Hold Ctrl or Shift to select multiple images.\n")

    filetypes = [
        ("All Supported Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tiff"),
        ("JPEG Images", "*.jpg;*.jpeg"),
        ("PNG Images", "*.png"),
        ("All Files", "*.*")
    ]

    selected_files = filedialog.askopenfilenames(
        title="Select Images for Panorama (Select Multiple Files)",
        filetypes=filetypes
    )

    root.destroy()

    files_list = list(selected_files)
    if not files_list:
        print("  [!] Warning: No image files were selected.")
    else:
        print(f"  [+] Selected {len(files_list)} images from File Explorer:")
        for idx, fpath in enumerate(files_list, 1):
            print(f"      {idx}. {os.path.basename(fpath)}")
        print()

    return files_list


def prompt_interactive_input() -> list:
    """Interactively prompts user to enter image file paths or open graphical file picker."""
    print("========================================")
    print("INTELLIGENT PANORAMA BUILDER")
    print("========================================")
    print("Select input method:")
    print("  [1] Open Graphical File Picker ('This PC' File Explorer window)")
    print("  [2] Enter/Paste image paths manually in terminal\n")

    try:
        choice = input("Enter choice (1 or 2, default is 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInteractive input cancelled.")
        sys.exit(0)

    if choice in ("", "1"):
        files = open_file_picker()
        if len(files) >= 2:
            return files
        print("Falling back to manual terminal path input...\n")

    print("Enter image paths one by one from any viewpoint/order in your panorama.")
    print("You can drag & drop image files into this terminal window or paste file paths.")
    print("Type 'done', 'build', or press Enter on an empty line when finished.\n")

    input_paths = []
    idx = 1
    while True:
        try:
            line = input(f"Image {idx} path (or 'done'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInteractive input cancelled.")
            sys.exit(0)

        if not line or line.lower() in ("done", "build", "exit", "quit"):
            if len(input_paths) < 2:
                print("  [!] Error: At least 2 images are required to build a panorama. Please enter more paths.\n")
                continue
            break

        cleaned_path = line.strip("'\"")
        if not os.path.exists(cleaned_path):
            print(f"  [!] Warning: File '{cleaned_path}' does not exist. Please check the path and try again.\n")
            continue

        input_paths.append(cleaned_path)
        print(f"  [+] Added: {os.path.basename(cleaned_path)} ({len(input_paths)} images total)\n")
        idx += 1

    print(f"\nCollected {len(input_paths)} images. Enabling automatic sequence reordering and building panorama...\n")
    return input_paths


def main():
    """Main CLI entrypoint."""
    parser = build_cli_parser()
    args = parser.parse_args()

    input_files = args.input
    input_dir = args.input_dir

    if args.no_auto_order:
        auto_order = False
    elif args.auto_order:
        auto_order = True
    else:
        auto_order = True

    if args.gui:
        input_files = open_file_picker()
        if not input_files:
            sys.exit(1)
        auto_order = True

    elif args.interactive or (not input_files and not input_dir):
        input_files = prompt_interactive_input()
        if not input_files:
            sys.exit(1)
        auto_order = True

    try:
        stitcher = PanoramaStitcher(
            feature_algorithm=args.feature,
            ratio_threshold=args.ratio,
            max_dimension=args.max_dimension,
            auto_order=auto_order,
            debug_mode=args.debug
        )

        stitcher.stitch(
            input_files=input_files,
            input_dir=input_dir,
            output_filepath=args.output
        )
        sys.exit(0)

    except PanoramaError as e:
        print(f"\n========================================", file=sys.stderr)
        print(f"PANORAMA BUILDER ERROR", file=sys.stderr)
        print(f"========================================", file=sys.stderr)
        print(f"{str(e)}", file=sys.stderr)
        print(f"========================================\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
