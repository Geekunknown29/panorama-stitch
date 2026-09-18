"""
Generator script to create reproducible, feature-rich sample image sets
for evaluation of the Intelligent Panorama Builder CLI tool.
"""

import os
import cv2
import numpy as np

def generate_sample_dataset():
    sample_dir = os.path.join("data", "sample")
    failure_dir = os.path.join("data", "failure_cases")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    # Composite wide scene dimensions
    W, H = 2400, 1000
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # 1. Sky Gradient Background
    y_coords = np.linspace(0, 1, H).reshape(H, 1, 1)
    sky = np.zeros((H, W, 3), dtype=np.float32)
    top_color = np.array([240, 180, 100], dtype=np.float32)
    bottom_color = np.array([120, 60, 20], dtype=np.float32)
    for c in range(3):
        sky[:, :, c] = top_color[c] * (1 - y_coords[:, :, 0]) + bottom_color[c] * y_coords[:, :, 0]
    canvas = sky.astype(np.uint8)

    # 2. Ground plane
    ground_y = int(H * 0.6)
    cv2.rectangle(canvas, (0, ground_y), (W, H), (40, 80, 40), -1)

    # 3. Architectural structures (rich feature points)
    np.random.seed(42)
    for i in range(15):
        bx = 80 + i * 150
        bw = 120
        bh = 300 + int(180 * np.sin(i * 0.8))
        by = ground_y - bh
        color = (int((i * 50) % 200 + 40), int((i * 80) % 200 + 40), int((i * 110) % 200 + 40))
        cv2.rectangle(canvas, (bx, by), (bx + bw, ground_y), color, -1)

        # High-contrast window grid
        for wx in range(bx + 15, bx + bw - 15, 22):
            for wy in range(by + 20, ground_y - 20, 28):
                wcolor = (255, 255, 200) if (wx + wy) % 2 == 0 else (40, 40, 40)
                cv2.rectangle(canvas, (wx, wy), (wx + 14, wy + 18), wcolor, -1)

    # 4. Text and geometric landmarks
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "ACADEMIC COMPUTER VISION PANORAMA BUILDER DEMO", (100, 120), font, 1.6, (255, 255, 255), 4)
    cv2.putText(canvas, "FEATURE STITCHING PIPELINE - LEFT - CENTER - RIGHT", (250, 200), font, 1.3, (0, 255, 255), 3)

    for k in range(90):
        cx = np.random.randint(50, W - 50)
        cy = np.random.randint(50, ground_y - 50)
        rad = np.random.randint(12, 38)
        col = tuple(map(int, np.random.randint(50, 255, 3)))
        cv2.circle(canvas, (cx, cy), rad, col, 2)
        cv2.line(canvas, (cx - rad, cy), (cx + rad, cy), col, 2)
        cv2.line(canvas, (cx, cy - rad), (cx, cy + rad), col, 2)

    # 5. Crop sequential overlapping views (v_width=1100, step=550, overlap=550px)
    v_width = 1100
    step = 550
    crops = []
    for i in range(6):
        x_start = i * step
        x_end = x_start + v_width
        if x_end > W:
            break
        crops.append(np.ascontiguousarray(canvas[:, x_start:x_end].copy()))

    # Apply subtle homography transform to simulate camera movement
    H_warp2 = np.array([
        [1.002, 0.001, -2.0],
        [-0.001, 0.998, 4.0],
        [0.000002, -0.000002, 1.0]
    ], dtype=np.float32)

    H_warp3 = np.array([
        [0.998, -0.001, 3.0],
        [0.001, 1.002, -3.0],
        [-0.000002, 0.000002, 1.0]
    ], dtype=np.float32)
    H_warp4 = np.array([
        [1.001, 0.002, -1.0],
        [-0.002, 0.999, 2.0],
        [0.000001, -0.000001, 1.0]
    ], dtype=np.float32)
    H_warp5 = np.array([
        [0.999, -0.002, 2.0],
        [0.002, 1.001, -1.0],
        [-0.000001, 0.000001, 1.0]
    ], dtype=np.float32)
    H_warp6 = np.array([
        [1.003, 0.0, -3.0],
        [0.0, 0.997, 3.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    warp_matrices = [H_warp2, H_warp3, H_warp4, H_warp5, H_warp6]
    sample_images = [crops[0]]
    for idx in range(1, len(crops)):
        H_w = warp_matrices[(idx - 1) % len(warp_matrices)]
        sample_images.append(cv2.warpPerspective(crops[idx], H_w, (v_width, H)))

    sample_paths = []
    for idx, img in enumerate(sample_images, 1):
        p = os.path.join(sample_dir, f"scene1_{idx:02d}.jpg")
        cv2.imwrite(p, img)
        sample_paths.append(p)
        print(f"Saved: {p}")

    # 6. Generate failure cases (unrelated scenes)
    np.random.seed(101)
    unrelated1 = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    unrelated2 = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(unrelated2, "NO MATCHABLE FEATURES HERE", (80, 300), font, 1.5, (255, 255, 255), 3)

    pf1 = os.path.join(failure_dir, "unrelated_01.jpg")
    pf2 = os.path.join(failure_dir, "unrelated_02.jpg")

    cv2.imwrite(pf1, unrelated1)
    cv2.imwrite(pf2, unrelated2)

    print(f"Saved: {pf1}")
    print(f"Saved: {pf2}")

if __name__ == "__main__":
    generate_sample_dataset()
