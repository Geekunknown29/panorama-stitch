"""
Automatic Image Ordering Module for Intelligent Panorama Builder.

Builds a RANSAC-validated image connectivity graph, detects disconnected components,
and constructs the most plausible spatial sequence using pairwise geometric consistency
and robust pair-quality scoring (not filename order).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from src.preprocessing import ProcessedImage
from src.feature_detector import FeatureDetector
from src.feature_matcher import FeatureMatcher
from src.homography import HomographyEstimator, PairQualityReport, ValidationConfig
from src.input_handler import PanoramaError


class OrderingError(PanoramaError):
    """Exception raised when image ordering fails."""
    pass


@dataclass
class OrderingResult:
    """Result of automatic image ordering."""
    ordered_images: List[ProcessedImage]
    ordered_indices: List[int]
    excluded_images: List[ProcessedImage]
    excluded_filenames: List[str]
    pair_reports: Dict[Tuple[int, int], PairQualityReport]
    pairwise_results: Dict[Tuple[int, int], Tuple[Any, Any, bool]] = field(default_factory=dict)
    disconnected: bool = False


class ImageOrderer:
    """Determines spatial image sequence using robust pair-quality geometric matching."""

    def __init__(
        self,
        detector: FeatureDetector,
        matcher: FeatureMatcher,
        config: Optional[ValidationConfig] = None,
        min_inliers: int = 15,
        min_quality_score: float = 0.45
    ):
        self.detector = detector
        self.matcher = matcher
        self.config = config or ValidationConfig()
        self.homography_estimator = HomographyEstimator(
            ransac_reproj_threshold=4.0,
            config=self.config
        )
        self.min_inliers = min_inliers
        self.min_quality_score = min_quality_score

    def find_optimal_order(
        self,
        images: List[ProcessedImage],
        precomputed_features: Optional[List[Tuple]] = None,
        allow_partial: bool = True
    ) -> OrderingResult:
        """
        Determines the optimal spatial sequence using pairwise geometric quality.

        Does NOT use filename order, timestamps, or numeric suffixes.
        """
        n = len(images)
        if n <= 2:
            return OrderingResult(
                ordered_images=list(images),
                ordered_indices=list(range(n)),
                excluded_images=[],
                excluded_filenames=[],
                pair_reports={},
                pairwise_results={},
            )

        if precomputed_features is not None and len(precomputed_features) == n:
            features = precomputed_features
        else:
            features = []
            for img in images:
                try:
                    kp, des = self.detector.detect_and_compute(img.gray_img)
                    features.append((kp, des))
                except Exception:
                    features.append(([], None))

        # adjacency[i][j] = quality-weighted edge score (0 = no reliable edge)
        adjacency = np.zeros((n, n), dtype=float)
        pairwise_info: Dict[Tuple[int, int], dict] = {}
        pair_reports: Dict[Tuple[int, int], PairQualityReport] = {}
        pairwise_results: Dict[Tuple[int, int], Tuple[Any, Any, bool]] = {}

        print(f"\n--- Automatic Image Ordering ({n} images) ---")
        print(f"Computing pairwise matches ({n * (n - 1) // 2} pairs)...\n")

        for i in range(n):
            for j in range(i + 1, n):
                kp_i, des_i = features[i]
                kp_j, des_j = features[j]

                result_is_lower_to_upper = True
                result = self.homography_estimator.estimate_pairwise(
                    kp_src=kp_i, des_src=des_i,
                    kp_dst=kp_j, des_dst=des_j,
                    matcher=self.matcher,
                    src_name=images[i].filename,
                    dst_name=images[j].filename,
                    src_shape=images[i].color_img.shape[:2],
                    dst_shape=images[j].color_img.shape[:2],
                    min_good_matches=8,
                    min_inliers=self.min_inliers
                )
                if result is None:
                    result_is_lower_to_upper = False
                    result = self.homography_estimator.estimate_pairwise(
                        kp_src=kp_j, des_src=des_j,
                        kp_dst=kp_i, des_dst=des_i,
                        matcher=self.matcher,
                        src_name=images[j].filename,
                        dst_name=images[i].filename,
                        src_shape=images[j].color_img.shape[:2],
                        dst_shape=images[i].color_img.shape[:2],
                        min_good_matches=8,
                        min_inliers=self.min_inliers
                    )

                if result is None:
                    continue

                homo_res, match_res, report = result
                if report is None or report.decision != "ACCEPT":
                    continue
                if report.quality_score < self.min_quality_score:
                    continue

                edge_score = report.quality_score * report.inliers_count
                adjacency[i][j] = edge_score
                adjacency[j][i] = edge_score
                pair_reports[(i, j)] = report
                pairwise_results[(i, j)] = (homo_res, match_res, result_is_lower_to_upper)
                pairwise_info[(i, j)] = {
                    "inliers": homo_res.inliers_count,
                    "inlier_ratio": homo_res.inlier_ratio,
                    "good_matches": match_res.good_matches_count,
                    "reproj_error": homo_res.reprojection_error,
                    "quality_score": report.quality_score,
                }
                print(
                    f"  {images[i].filename} ↔ {images[j].filename}: "
                    f"quality={report.quality_score:.3f}, "
                    f"{homo_res.inliers_count} inliers ({homo_res.inlier_ratio:.1f}%), "
                    f"spatial={report.spatial.area_coverage_pct:.0f}%"
                )

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i][j] > 0:
                    union(i, j)

        components: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            components.setdefault(root, []).append(i)

        if len(components) > 1:
            if not allow_partial:
                comp_strs = []
                for comp_nodes in components.values():
                    names = [images[idx].filename for idx in comp_nodes]
                    comp_strs.append(", ".join(names))
                detail = "\n".join(f"  Group {k + 1}: {s}" for k, s in enumerate(comp_strs))
                raise OrderingError(
                    f"Unable to create a single panorama because some images do not have "
                    f"sufficient overlap.\n"
                    f"Found {len(components)} disconnected groups:\n{detail}\n"
                    f"Each group's images overlap with each other but not with images "
                    f"in other groups."
                )

            largest_root = max(components.keys(), key=lambda r: len(components[r]))
            connected_nodes = components[largest_root]
            excluded_indices = [i for i in range(n) if i not in connected_nodes]
            excluded = [images[i] for i in excluded_indices]
            excluded_names = [images[i].filename for i in excluded_indices]

            print(
                f"\nSome images do not have sufficient reliable overlap and were excluded:"
            )
            for name in excluded_names:
                print(f"  - {name}")
            print(
                f"\n{len(connected_nodes)} images successfully connected."
            )
        else:
            connected_nodes = list(components.values())[0]
            excluded = []
            excluded_names = []

        chain = self._find_best_chain(adjacency, connected_nodes)

        print(f"\nDetected panorama order:")
        for k, idx in enumerate(chain):
            prefix = "  " if k == 0 else "  "
            arrow = "" if k == 0 else f"{k + 1}. "
            print(f"{prefix}{arrow}{images[idx].filename}")
        print()

        ordered_images = [images[idx] for idx in chain]
        return OrderingResult(
            ordered_images=ordered_images,
            ordered_indices=chain,
            excluded_images=excluded,
            excluded_filenames=excluded_names,
            pair_reports=pair_reports,
            pairwise_results=pairwise_results,
            disconnected=len(components) > 1,
        )

    def _find_best_chain(
        self,
        adjacency: np.ndarray,
        nodes: List[int]
    ) -> List[int]:
        n_nodes = len(nodes)
        if n_nodes <= 2:
            return nodes

        degrees = {}
        for node in nodes:
            deg = sum(1 for other in nodes if other != node and adjacency[node][other] > 0)
            degrees[node] = deg

        endpoints = [nd for nd in nodes if degrees[nd] == 1]
        if not endpoints:
            endpoints = sorted(nodes, key=lambda nd: degrees[nd])

        best_chain = []
        best_score = -1.0

        for start_node in endpoints:
            chain = self._find_best_hamiltonian_path(adjacency, nodes, start_node)
            if not chain:
                chain = self._greedy_chain_from(adjacency, nodes, start_node)
            score = self._chain_score(adjacency, chain)
            if len(chain) > len(best_chain) or (len(chain) == len(best_chain) and score > best_score):
                best_chain = chain
                best_score = score
            if len(best_chain) == n_nodes and best_score > 0:
                break

        if len(best_chain) < n_nodes:
            remaining = [nd for nd in nodes if nd not in best_chain]
            for node in remaining:
                left_score = adjacency[best_chain[0]][node] if best_chain else 0
                right_score = adjacency[best_chain[-1]][node] if best_chain else 0
                if left_score >= right_score and left_score > 0:
                    best_chain.insert(0, node)
                elif right_score > 0:
                    best_chain.append(node)
                else:
                    best_chain.append(node)

        return best_chain

    @staticmethod
    def _chain_score(adjacency: np.ndarray, chain: List[int]) -> float:
        if len(chain) < 2:
            return 0.0
        return float(sum(adjacency[chain[i]][chain[i + 1]] for i in range(len(chain) - 1)))

    def _find_best_hamiltonian_path(
        self,
        adjacency: np.ndarray,
        nodes: List[int],
        start: int
    ) -> List[int]:
        """DFS search for highest-scoring path through the connectivity graph."""
        node_set = set(nodes)
        best_path: List[int] = []
        best_score = -1.0

        def dfs(current: int, visited: List[int], score: float):
            nonlocal best_path, best_score
            if len(visited) > len(best_path) or (
                len(visited) == len(best_path) and score > best_score
            ):
                best_path = list(visited)
                best_score = score
            if len(visited) == len(nodes):
                return
            for nxt in node_set:
                if nxt in visited:
                    continue
                edge = adjacency[current][nxt]
                if edge <= 0:
                    continue
                dfs(nxt, visited + [nxt], score + edge)

        dfs(start, [start], 0.0)
        return best_path

    def _greedy_chain_from(
        self,
        adjacency: np.ndarray,
        nodes: List[int],
        start: int
    ) -> List[int]:
        node_set = set(nodes)
        visited = {start}
        chain = [start]

        while len(visited) < len(nodes):
            extended = False

            right_end = chain[-1]
            best_right = -1
            best_right_score = 0.0
            for cand in node_set - visited:
                score = adjacency[right_end][cand]
                if score > best_right_score:
                    best_right_score = score
                    best_right = cand

            left_end = chain[0]
            best_left = -1
            best_left_score = 0.0
            for cand in node_set - visited:
                score = adjacency[left_end][cand]
                if score > best_left_score:
                    best_left_score = score
                    best_left = cand

            if best_right_score >= best_left_score and best_right != -1 and best_right_score > 0:
                chain.append(best_right)
                visited.add(best_right)
                extended = True
            elif best_left != -1 and best_left_score > 0:
                chain.insert(0, best_left)
                visited.add(best_left)
                extended = True

            if not extended:
                break

        return chain
