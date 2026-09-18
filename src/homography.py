"""
Homography Estimator Module for Intelligent Panorama Builder.

Estimates 3x3 homography transformation matrix using OpenCV findHomography and RANSAC,
rejecting geometric outliers, validating corner projection geometry, spatial inlier
distribution, and stabilizing multi-image alignments.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np
from src.input_handler import PanoramaError


class HomographyEstimationError(PanoramaError):
    """Exception raised when homography estimation fails or produces degenerate transformation."""
    pass


@dataclass
class ValidationConfig:
    """
    Configurable thresholds for homography and pair-quality validation.

    Threshold rationale (panorama stitching with working copies up to ~1600 px):
    - Area/scale limits reject runaway perspective that blows up canvas size.
    - Corner displacement limits keep projected geometry near plausible overlap.
    - Spatial coverage rejects homographies fit on a tiny feature cluster.
    - Reprojection percentiles catch outliers that a low mean can hide.
    """
    min_inliers: int = 10
    min_inlier_ratio_pct: float = 15.0
    max_mean_reproj_error: float = 5.0
    max_median_reproj_error: float = 4.0
    max_p95_reproj_error: float = 10.0
    max_max_reproj_error: float = 20.0
    min_spatial_x_coverage: float = 0.20
    min_spatial_y_coverage: float = 0.20
    min_spatial_area_coverage: float = 0.10
    min_quadrants_with_inliers: int = 2
    min_quadrant_inlier_fraction: float = 0.05
    min_area_ratio: float = 0.05
    max_area_ratio: float = 4.0
    min_scale_change: float = 0.25
    max_scale_change: float = 3.0
    max_projected_dimension: int = 8000
    max_corner_displacement_ratio: float = 2.5
    max_condition_number: float = 1e6
    min_determinant: float = 1e-8
    max_determinant: float = 1e6
    max_perspective_term: float = 5e-3
    min_accept_quality_score: float = 0.45


@dataclass
class ReprojectionMetrics:
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    max: float = 0.0


@dataclass
class SpatialCoverageMetrics:
    bbox_x_min: float = 0.0
    bbox_x_max: float = 0.0
    bbox_y_min: float = 0.0
    bbox_y_max: float = 0.0
    x_coverage_pct: float = 0.0
    y_coverage_pct: float = 0.0
    area_coverage_pct: float = 0.0
    convex_hull_area_pct: float = 0.0
    quadrant_counts: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    quadrants_with_inliers: int = 0


@dataclass
class ProjectedGeometryMetrics:
    corners: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))
    width: float = 0.0
    height: float = 0.0
    area: float = 0.0
    area_ratio: float = 0.0
    scale_change: float = 1.0
    is_convex: bool = False
    max_corner_displacement: float = 0.0
    perspective_distortion: float = 0.0


@dataclass
class NumericalMetrics:
    determinant: float = 0.0
    condition_number: float = 0.0
    is_finite: bool = True
    perspective_h31: float = 0.0
    perspective_h32: float = 0.0


@dataclass
class PairQualityReport:
    pair_name: str = ""
    good_matches: int = 0
    inliers_count: int = 0
    inlier_ratio: float = 0.0
    reprojection: ReprojectionMetrics = field(default_factory=ReprojectionMetrics)
    spatial: SpatialCoverageMetrics = field(default_factory=SpatialCoverageMetrics)
    geometry: ProjectedGeometryMetrics = field(default_factory=ProjectedGeometryMetrics)
    numerical: NumericalMetrics = field(default_factory=NumericalMetrics)
    quality_score: float = 0.0
    decision: str = "REJECT"
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair_name,
            "good_matches": self.good_matches,
            "inliers": self.inliers_count,
            "inlier_ratio_pct": round(self.inlier_ratio, 2),
            "reprojection": {
                "mean": round(self.reprojection.mean, 4),
                "median": round(self.reprojection.median, 4),
                "p95": round(self.reprojection.p95, 4),
                "max": round(self.reprojection.max, 4),
            },
            "spatial_coverage": {
                "x_pct": round(self.spatial.x_coverage_pct, 2),
                "y_pct": round(self.spatial.y_coverage_pct, 2),
                "area_pct": round(self.spatial.area_coverage_pct, 2),
                "convex_hull_area_pct": round(self.spatial.convex_hull_area_pct, 2),
                "quadrants_with_inliers": self.spatial.quadrants_with_inliers,
            },
            "projected_geometry": {
                "width": round(self.geometry.width, 1),
                "height": round(self.geometry.height, 1),
                "area_ratio": round(self.geometry.area_ratio, 3),
                "scale_change": round(self.geometry.scale_change, 3),
                "is_convex": self.geometry.is_convex,
                "max_corner_displacement": round(self.geometry.max_corner_displacement, 1),
            },
            "numerical": {
                "determinant": f"{self.numerical.determinant:.4e}",
                "condition_number": f"{self.numerical.condition_number:.4e}",
                "is_finite": self.numerical.is_finite,
            },
            "quality_score": round(self.quality_score, 3),
            "decision": self.decision,
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass
class HomographyResult:
    """Dataclass storing estimated 3x3 homography matrix, RANSAC inlier statistics, and metrics."""
    matrix: np.ndarray
    inliers_mask: np.ndarray
    inliers_count: int
    outliers_count: int
    inlier_ratio: float
    reprojection_error: float
    transform_type: str = "homography"
    quality_report: Optional[PairQualityReport] = None


class HomographyEstimator:
    """Manages homography estimation, geometric sanity validation, and transformation stabilization."""

    def __init__(
        self,
        ransac_reproj_threshold: float = 4.0,
        max_iters: int = 2000,
        config: Optional[ValidationConfig] = None
    ):
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.max_iters = max_iters
        self.config = config or ValidationConfig()

    @staticmethod
    def project_corners(img_shape: Tuple[int, int], H: np.ndarray) -> np.ndarray:
        """Projects TL, BL, BR, TR corners through H (OpenCV corner order)."""
        h, w = img_shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H)
        return proj.reshape(4, 2)

    @staticmethod
    def compute_reprojection_metrics(
        H: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        inlier_mask: np.ndarray
    ) -> ReprojectionMetrics:
        inlier_indices = np.where(inlier_mask.ravel() == 1)[0]
        if len(inlier_indices) == 0:
            return ReprojectionMetrics()

        src_inliers = src_pts[inlier_indices].reshape(-1, 2)
        dst_inliers = dst_pts[inlier_indices].reshape(-1, 2)
        src_homo = np.hstack([src_inliers, np.ones((len(src_inliers), 1))])
        projected_homo = (H @ src_homo.T).T
        projected = projected_homo[:, :2] / projected_homo[:, 2:3]
        errors = np.linalg.norm(projected - dst_inliers, axis=1)

        return ReprojectionMetrics(
            mean=float(np.mean(errors)),
            median=float(np.median(errors)),
            p95=float(np.percentile(errors, 95)),
            max=float(np.max(errors)),
        )

    @staticmethod
    def compute_spatial_coverage(
        src_pts: np.ndarray,
        inlier_mask: np.ndarray,
        src_shape: Tuple[int, int]
    ) -> SpatialCoverageMetrics:
        h, w = src_shape[:2]
        inlier_indices = np.where(inlier_mask.ravel() == 1)[0]
        if len(inlier_indices) == 0:
            return SpatialCoverageMetrics()

        pts = src_pts[inlier_indices].reshape(-1, 2)
        x_min, y_min = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
        x_max, y_max = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))

        bbox_w = max(x_max - x_min, 1.0)
        bbox_h = max(y_max - y_min, 1.0)
        x_cov = min(bbox_w / float(w), 1.0) * 100.0
        y_cov = min(bbox_h / float(h), 1.0) * 100.0
        area_cov = min((bbox_w * bbox_h) / float(w * h), 1.0) * 100.0
        hull = cv2.convexHull(pts.astype(np.float32)) if len(pts) >= 3 else None
        hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
        hull_area_cov = min(hull_area / float(w * h), 1.0) * 100.0

        mid_x, mid_y = w / 2.0, h / 2.0
        quadrants = [0, 0, 0, 0]
        for x, y in pts:
            qx = 0 if x < mid_x else 1
            qy = 0 if y < mid_y else 1
            quadrants[qy * 2 + qx] += 1

        total = len(pts)
        min_frac = 0.05
        quadrants_with = sum(1 for c in quadrants if c / total >= min_frac)

        return SpatialCoverageMetrics(
            bbox_x_min=x_min, bbox_x_max=x_max,
            bbox_y_min=y_min, bbox_y_max=y_max,
            x_coverage_pct=x_cov, y_coverage_pct=y_cov,
            area_coverage_pct=area_cov,
            convex_hull_area_pct=hull_area_cov,
            quadrant_counts=quadrants,
            quadrants_with_inliers=quadrants_with,
        )

    @staticmethod
    def compute_projected_geometry(
        H: np.ndarray,
        src_shape: Tuple[int, int],
        dst_shape: Optional[Tuple[int, int]] = None
    ) -> ProjectedGeometryMetrics:
        h, w = src_shape[:2]
        orig_area = float(w * h)
        corners = HomographyEstimator.project_corners(src_shape, H)

        is_finite = bool(np.all(np.isfinite(corners)))
        is_convex = bool(cv2.isContourConvex(np.int32(corners))) if is_finite else False
        proj_area = float(cv2.contourArea(corners)) if is_finite and is_convex else 0.0
        area_ratio = proj_area / orig_area if orig_area > 0 else 0.0
        scale_change = float(np.sqrt(max(area_ratio, 0.0)))

        x_coords, y_coords = corners[:, 0], corners[:, 1]
        proj_w = float(np.max(x_coords) - np.min(x_coords))
        proj_h = float(np.max(y_coords) - np.min(y_coords))

        max_disp = 0.0
        if dst_shape is not None:
            dh, dw = dst_shape[:2]
            ref_dim = max(w, h, dw, dh)
            dst_rect = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float64)
            for corner in corners:
                dists = np.linalg.norm(dst_rect - corner, axis=1)
                max_disp = max(max_disp, float(np.min(dists)))
        else:
            ref_dim = max(w, h)

        edge_lengths = []
        for i in range(4):
            p1, p2 = corners[i], corners[(i + 1) % 4]
            edge_lengths.append(float(np.linalg.norm(p2 - p1)))
        if len(edge_lengths) >= 4:
            top, left, bottom, right = edge_lengths[0], edge_lengths[3], edge_lengths[2], edge_lengths[1]
            perspective = max(top, bottom) / max(min(top, bottom), 1.0)
            perspective = max(perspective, max(left, right) / max(min(left, right), 1.0))
        else:
            perspective = 1.0

        return ProjectedGeometryMetrics(
            corners=corners,
            width=proj_w,
            height=proj_h,
            area=proj_area,
            area_ratio=area_ratio,
            scale_change=scale_change,
            is_convex=is_convex,
            max_corner_displacement=max_disp,
            perspective_distortion=perspective,
        )

    @staticmethod
    def compute_numerical_metrics(H: np.ndarray) -> NumericalMetrics:
        is_finite = bool(np.all(np.isfinite(H)))
        det = float(np.linalg.det(H)) if is_finite else 0.0
        cond = 0.0
        if is_finite:
            sv = np.linalg.svd(H, compute_uv=False)
            cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-15 else float("inf")
        return NumericalMetrics(
            determinant=det,
            condition_number=cond,
            is_finite=is_finite,
            perspective_h31=float(H[2, 0]) if is_finite else 0.0,
            perspective_h32=float(H[2, 1]) if is_finite else 0.0,
        )

    def evaluate_pair_quality(
        self,
        H: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        inlier_mask: np.ndarray,
        good_matches: int,
        pair_name: str,
        src_shape: Tuple[int, int],
        dst_shape: Optional[Tuple[int, int]] = None,
        strict: bool = True
    ) -> PairQualityReport:
        cfg = self.config
        inliers_count = int(np.sum(inlier_mask))
        inlier_ratio = (inliers_count / float(len(inlier_mask))) * 100.0 if len(inlier_mask) > 0 else 0.0

        reproj = self.compute_reprojection_metrics(H, src_pts, dst_pts, inlier_mask)
        spatial = self.compute_spatial_coverage(src_pts, inlier_mask, src_shape)
        geometry = self.compute_projected_geometry(H, src_shape, dst_shape)
        numerical = self.compute_numerical_metrics(H)

        reasons: List[str] = []

        if good_matches < cfg.min_inliers:
            reasons.append(f"Too few good matches ({good_matches} < {cfg.min_inliers}).")
        if inliers_count < cfg.min_inliers:
            reasons.append(f"Too few RANSAC inliers ({inliers_count} < {cfg.min_inliers}).")
        if inlier_ratio < cfg.min_inlier_ratio_pct:
            reasons.append(f"Inlier ratio too low ({inlier_ratio:.1f}% < {cfg.min_inlier_ratio_pct}%).")
        if reproj.mean > cfg.max_mean_reproj_error:
            reasons.append(f"Mean reprojection error too high ({reproj.mean:.2f}px).")
        if reproj.median > cfg.max_median_reproj_error:
            reasons.append(f"Median reprojection error too high ({reproj.median:.2f}px).")
        if reproj.p95 > cfg.max_p95_reproj_error:
            reasons.append(f"95th percentile reprojection error too high ({reproj.p95:.2f}px).")
        if reproj.max > cfg.max_max_reproj_error:
            reasons.append(f"Maximum reprojection error too high ({reproj.max:.2f}px).")
        sufficient_spatial_coverage = (
            (
                spatial.x_coverage_pct >= 18.0
                and spatial.y_coverage_pct >= 45.0
            )
            or spatial.convex_hull_area_pct >= 12.0
            or spatial.quadrants_with_inliers >= cfg.min_quadrants_with_inliers
        )
        if not sufficient_spatial_coverage:
            reasons.append(
                f"Inliers have insufficient spatial distribution "
                f"(bbox={spatial.x_coverage_pct:.1f}% x {spatial.y_coverage_pct:.1f}%, "
                f"hull={spatial.convex_hull_area_pct:.1f}%, "
                f"quadrants={spatial.quadrants_with_inliers})."
            )
        if not numerical.is_finite:
            reasons.append("Homography matrix contains non-finite values.")
        if abs(numerical.determinant) < cfg.min_determinant:
            reasons.append(f"Degenerate determinant ({numerical.determinant:.2e}).")
        if numerical.determinant < 0:
            reasons.append("Negative determinant (flipped/wrong orientation).")
        if abs(numerical.determinant) > cfg.max_determinant:
            reasons.append(f"Extreme determinant ({numerical.determinant:.2e}).")
        if numerical.condition_number > cfg.max_condition_number:
            reasons.append(f"Ill-conditioned matrix (cond={numerical.condition_number:.2e}).")
        if abs(numerical.perspective_h31) > cfg.max_perspective_term:
            reasons.append(f"Excessive perspective term H[2,0]={numerical.perspective_h31:.2e}.")
        if abs(numerical.perspective_h32) > cfg.max_perspective_term:
            reasons.append(f"Excessive perspective term H[2,1]={numerical.perspective_h32:.2e}.")
        if not geometry.is_convex:
            reasons.append("Projected corners do not form a convex quadrilateral.")
        if geometry.area_ratio < cfg.min_area_ratio or geometry.area_ratio > cfg.max_area_ratio:
            reasons.append(f"Projected area ratio out of range ({geometry.area_ratio:.2f}).")
        if geometry.scale_change < cfg.min_scale_change or geometry.scale_change > cfg.max_scale_change:
            reasons.append(f"Scale change out of range ({geometry.scale_change:.2f}).")
        if geometry.width > cfg.max_projected_dimension or geometry.height > cfg.max_projected_dimension:
            reasons.append(
                f"Projected dimensions too large ({geometry.width:.0f} x {geometry.height:.0f})."
            )

        ref_dim = max(src_shape[1], src_shape[0])
        if dst_shape is not None:
            ref_dim = max(ref_dim, dst_shape[1], dst_shape[0])
        max_allowed_disp = cfg.max_corner_displacement_ratio * ref_dim
        if geometry.max_corner_displacement > max_allowed_disp:
            reasons.append(
                f"Projected corners too far from destination frame "
                f"(displacement={geometry.max_corner_displacement:.0f}px, "
                f"limit={max_allowed_disp:.0f}px)."
            )

        quality_score = self._compute_quality_score(
            good_matches, inlier_ratio, reproj, spatial, geometry, numerical
        )

        decision = "REJECT"
        if not reasons:
            if quality_score >= cfg.min_accept_quality_score:
                decision = "ACCEPT"
            else:
                reasons.append(
                    f"Quality score too low ({quality_score:.3f} < {cfg.min_accept_quality_score})."
                )
        elif not strict and quality_score >= 0.65 and inliers_count >= cfg.min_inliers:
            decision = "ACCEPT"

        return PairQualityReport(
            pair_name=pair_name,
            good_matches=good_matches,
            inliers_count=inliers_count,
            inlier_ratio=inlier_ratio,
            reprojection=reproj,
            spatial=spatial,
            geometry=geometry,
            numerical=numerical,
            quality_score=quality_score,
            decision=decision,
            rejection_reasons=reasons,
        )

    @staticmethod
    def _compute_quality_score(
        good_matches: int,
        inlier_ratio: float,
        reproj: ReprojectionMetrics,
        spatial: SpatialCoverageMetrics,
        geometry: ProjectedGeometryMetrics,
        numerical: NumericalMetrics
    ) -> float:
        match_score = min(good_matches / 200.0, 1.0)
        inlier_score = min(inlier_ratio / 80.0, 1.0)
        reproj_score = max(0.0, 1.0 - reproj.median / 5.0)
        spatial_score = (
            (spatial.x_coverage_pct / 100.0) * 0.33
            + (spatial.y_coverage_pct / 100.0) * 0.33
            + (spatial.area_coverage_pct / 100.0) * 0.34
        )
        scale_penalty = 0.0
        if geometry.scale_change < 0.25 or geometry.scale_change > 3.0:
            scale_penalty = 0.5
        elif geometry.scale_change < 0.4 or geometry.scale_change > 2.5:
            scale_penalty = 0.2
        geom_score = 1.0 - scale_penalty
        if not geometry.is_convex:
            geom_score *= 0.2
        num_score = 1.0
        if not numerical.is_finite or numerical.determinant <= 0:
            num_score = 0.0
        elif numerical.condition_number > 1e5:
            num_score = 0.3

        return float(
            match_score * 0.15
            + inlier_score * 0.20
            + reproj_score * 0.20
            + spatial_score * 0.25
            + geom_score * 0.15
            + num_score * 0.05
        )

    @staticmethod
    def validate_projected_corners(
        proj_corners: np.ndarray,
        orig_shape: Tuple[int, int],
        min_area_ratio: float = 0.05,
        max_area_ratio: float = 4.0
    ) -> Tuple[bool, str]:
        h, w = orig_shape[:2]
        orig_area = float(w * h)

        if not np.all(np.isfinite(proj_corners)):
            return False, "Projected coordinates contain NaN or Inf values."

        is_convex = cv2.isContourConvex(np.int32(proj_corners))
        if not is_convex:
            return False, "Projected corners do not form a convex quadrilateral (perspective distortion/folding)."

        proj_area = float(cv2.contourArea(proj_corners))
        area_ratio = proj_area / orig_area if orig_area > 0 else 0.0

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            return False, (
                f"Projected area ratio ({area_ratio:.2f}) is outside acceptable range "
                f"[{min_area_ratio}, {max_area_ratio}]."
            )

        return True, "Valid"

    def validate_homography(
        self,
        H: np.ndarray,
        src_shape: Tuple[int, int],
        dst_shape: Optional[Tuple[int, int]] = None,
        max_projected_dimension: Optional[int] = None
    ) -> Tuple[bool, str]:
        cfg = self.config
        max_dim = max_projected_dimension or cfg.max_projected_dimension

        if H is None:
            return False, "Homography matrix is None."

        numerical = self.compute_numerical_metrics(H)
        if not numerical.is_finite:
            return False, "Homography matrix contains non-finite values."
        if abs(numerical.determinant) < cfg.min_determinant:
            return False, f"Degenerate homography (determinant={numerical.determinant:.2e}, near zero)."
        if numerical.determinant < 0:
            return False, f"Flipped homography (determinant={numerical.determinant:.2e}, negative)."
        if abs(numerical.determinant) > cfg.max_determinant:
            return False, f"Extreme homography (determinant={numerical.determinant:.2e}, too large)."
        if numerical.condition_number > cfg.max_condition_number:
            return False, f"Ill-conditioned homography (condition number={numerical.condition_number:.2e})."

        geometry = self.compute_projected_geometry(H, src_shape, dst_shape)
        if not geometry.is_convex:
            return False, "Projected corners do not form a convex quadrilateral."

        is_valid, reason = self.validate_projected_corners(
            geometry.corners, src_shape,
            min_area_ratio=cfg.min_area_ratio,
            max_area_ratio=cfg.max_area_ratio
        )
        if not is_valid:
            return False, reason

        if geometry.width > max_dim or geometry.height > max_dim:
            return False, (
                f"Projected dimensions ({geometry.width:.0f} x {geometry.height:.0f}) exceed maximum "
                f"allowed dimension ({max_dim})."
            )

        if geometry.scale_change < cfg.min_scale_change or geometry.scale_change > cfg.max_scale_change:
            return False, f"Scale change ({geometry.scale_change:.2f}) outside acceptable range."

        ref_dim = max(src_shape[1], src_shape[0])
        if dst_shape is not None:
            ref_dim = max(ref_dim, dst_shape[1], dst_shape[0])
        if geometry.max_corner_displacement > cfg.max_corner_displacement_ratio * ref_dim:
            return False, (
                f"Projected corners too far from reference frame "
                f"(displacement={geometry.max_corner_displacement:.0f}px)."
            )

        return True, "Valid"

    def estimate(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
        pair_name: str = "img1_img2",
        min_inliers: int = 10,
        src_shape: Optional[Tuple[int, int]] = None,
        dst_shape: Optional[Tuple[int, int]] = None,
        good_matches: Optional[int] = None,
        validate: bool = True
    ) -> HomographyResult:
        """
        Estimates 3x3 homography matrix mapping src_pts to dst_pts.

        Convention: H maps coordinates FROM the source image frame TO the destination image frame.
        i.e. dst_pt ≈ H @ src_pt (in homogeneous coordinates).
        """
        if src_pts is None or dst_pts is None or len(src_pts) < 4:
            raise HomographyEstimationError(
                f"Homography estimation failed for pair '{pair_name}': "
                f"Requires at least 4 point correspondences, but got {0 if src_pts is None else len(src_pts)}."
            )

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
            maxIters=self.max_iters
        )

        if H is None or mask is None:
            raise HomographyEstimationError(
                f"ERROR:\nHomography estimation failed for image pair '{pair_name}'.\n"
                "RANSAC algorithm could not compute a valid 3x3 transformation matrix."
            )

        inliers_count = int(np.sum(mask))
        outliers_count = len(mask) - inliers_count
        inlier_ratio = (inliers_count / float(len(mask))) * 100.0 if len(mask) > 0 else 0.0
        gm = good_matches if good_matches is not None else len(mask)

        if inliers_count < min_inliers:
            raise HomographyEstimationError(
                f"ERROR:\nInsufficient RANSAC inliers for image pair '{pair_name}'.\n"
                f"Found: {inliers_count} inlier(s)\n"
                f"Required: {min_inliers} minimum inlier(s)\n"
                "The image pair may have poor geometric overlap or severe parallax."
            )

        quality_report = None
        if src_shape is not None:
            quality_report = self.evaluate_pair_quality(
                H, src_pts, dst_pts, mask, gm, pair_name, src_shape, dst_shape
            )
            if validate and quality_report.decision == "REJECT":
                reasons = "; ".join(quality_report.rejection_reasons)
                raise HomographyEstimationError(
                    f"ERROR:\nHomography rejected for '{pair_name}': {reasons}"
                )

        reproj = self.compute_reprojection_metrics(H, src_pts, dst_pts, mask)

        return HomographyResult(
            matrix=H,
            inliers_mask=mask,
            inliers_count=inliers_count,
            outliers_count=outliers_count,
            inlier_ratio=inlier_ratio,
            reprojection_error=reproj.mean,
            transform_type="homography",
            quality_report=quality_report,
        )

    def estimate_pairwise(
        self,
        kp_src, des_src,
        kp_dst, des_dst,
        matcher,
        src_name: str = "img_src",
        dst_name: str = "img_dst",
        src_shape: Optional[Tuple[int, int]] = None,
        dst_shape: Optional[Tuple[int, int]] = None,
        min_good_matches: int = 10,
        min_inliers: int = 10
    ) -> Optional[Tuple[HomographyResult, 'MatchResult', PairQualityReport]]:
        """
        Complete pairwise estimation: match → ratio test → RANSAC → validation.

        Convention: The returned H maps coordinates FROM src image TO dst image.
        """
        from src.feature_matcher import FeatureMatchingError

        if des_src is None or des_dst is None:
            return None
        if len(des_src) < 2 or len(des_dst) < 2:
            return None

        try:
            match_res = matcher.match(
                kp_src, des_src,
                kp_dst, des_dst,
                img1_name=src_name,
                img2_name=dst_name,
                min_good_matches=min_good_matches
            )
        except FeatureMatchingError:
            return None

        try:
            homo_res = self.estimate(
                src_pts=match_res.src_pts,
                dst_pts=match_res.dst_pts,
                pair_name=f"{src_name} → {dst_name}",
                min_inliers=min_inliers,
                src_shape=src_shape,
                dst_shape=dst_shape,
                good_matches=match_res.good_matches_count,
                validate=True
            )
        except HomographyEstimationError:
            return None

        report = homo_res.quality_report
        if report is None and src_shape is not None:
            report = self.evaluate_pair_quality(
                homo_res.matrix,
                match_res.src_pts,
                match_res.dst_pts,
                homo_res.inliers_mask,
                match_res.good_matches_count,
                f"{src_name} → {dst_name}",
                src_shape,
                dst_shape,
            )

        if report and report.decision == "REJECT":
            return None

        return homo_res, match_res, report

    @staticmethod
    def print_pair_quality_report(report: PairQualityReport) -> None:
        """Prints formatted pair quality diagnostics to terminal."""
        print(f"  Pair: {report.pair_name}")
        print(f"  Good matches: {report.good_matches}")
        print(f"  RANSAC inliers: {report.inliers_count}")
        print(f"  Inlier ratio: {report.inlier_ratio:.1f}%")
        print(f"  Median error: {report.reprojection.median:.3f} px")
        print(f"  95th percentile: {report.reprojection.p95:.3f} px")
        print(f"  Spatial coverage: X={report.spatial.x_coverage_pct:.1f}%, "
              f"Y={report.spatial.y_coverage_pct:.1f}%, "
              f"area={report.spatial.area_coverage_pct:.1f}%")
        print(f"  Projected dimensions: {report.geometry.width:.0f} x {report.geometry.height:.0f}")
        print(f"  Geometry: convex={report.geometry.is_convex}, "
              f"scale={report.geometry.scale_change:.2f}, "
              f"area_ratio={report.geometry.area_ratio:.2f}")
        print(f"  Quality: {report.quality_score:.3f}")
        print(f"  Decision: {report.decision}")
        if report.rejection_reasons:
            for reason in report.rejection_reasons:
                print(f"    - {reason}")
