import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ---------- procedural front faces -------------------------------------------------------

def plank_stack(rng, S):
    """Return (RGBA-like image, mask, boxes[(x0,y0,x1,y1)]) of a stack front, size ~S."""
    cols = int(rng.integers(1, 6))
    layers = int(rng.integers(6, 26))
    gap_c = int(rng.integers(3, 14))
    face_w = int(rng.integers(60, 130))
    thick = int(rng.integers(14, 28))
    W = cols * face_w + (cols - 1) * gap_c
    H = layers * thick
    sc = min(0.80 * S / W, 0.80 * S / H, 2.0)
    face_w, thick, gap_c = max(8, int(face_w * sc)), max(6, int(thick * sc)), max(2, int(gap_c * sc))
    W, H = cols * face_w + (cols - 1) * gap_c, layers * thick
    img = np.zeros((H, W, 3), np.float32)
    boxes = []
    base = rng.uniform(150, 215)
    holes_per = int(rng.choice([0, 1, 2, 2, 2]))
    for c in range(cols):
        x0 = c * (face_w + gap_c)
        ctone = base + rng.uniform(-18, 18)
        for l in range(layers):
            y0 = l * thick
            tone = ctone + rng.uniform(-14, 14)
            face = np.full((thick, face_w, 3), tone, np.float32)
            face += np.linspace(rng.uniform(-18, 18), rng.uniform(-18, 18), face_w)[None, :, None]
            face[:2] += 28                       # lit top edge
            face[-3:] -= rng.uniform(35, 75)     # shadow gap towards the plank below
            for k in range(holes_per):
                hx = int(face_w * (0.5 if holes_per == 1 else (0.28, 0.72)[k]) + rng.integers(-3, 4))
                cv2.circle(face, (hx, thick // 2), max(1, int(thick * rng.uniform(0.09, 0.15))), (25, 25, 25), -1, cv2.LINE_AA)
            img[y0:y0 + thick, x0:x0 + face_w] = face
            boxes.append((x0, y0, x0 + face_w, y0 + thick))
    img += rng.normal(0, 3, img.shape)
    mask = np.zeros((H, W), np.uint8)
    for c in range(cols):
        mask[:, c * (face_w + gap_c): c * (face_w + gap_c) + face_w] = 255
    return np.clip(img, 0, 255).astype(np.uint8), mask, boxes


def tube_bundle(rng, S):
    """Hex lattice of tube ends (dark bore, bright rim)."""
    rows = int(rng.integers(4, 11))
    cols = int(rng.integers(5, 13))
    r = int(rng.integers(8, 20))
    pitch = 2 * r + int(rng.integers(0, 3))
    W = cols * pitch + pitch // 2
    H = int(rows * pitch * 0.866) + pitch
    sc = min(0.9 * S / W, 0.9 * S / H, 2.0)
    r, pitch = max(4, int(r * sc)), max(9, int(pitch * sc))
    W, H = cols * pitch + pitch // 2, int(rows * pitch * 0.866) + pitch
    img = np.full((H, W, 3), rng.uniform(70, 150), np.float32)
    mask = np.zeros((H, W), np.uint8)
    boxes = []
    for i in range(rows):
        for j in range(cols):
            cx = int(j * pitch + pitch / 2 + (pitch / 2 if i % 2 else 0))
            cy = int(i * pitch * 0.866 + pitch / 2)
            if cx + r >= W or cy + r >= H:
                continue
            rim = rng.uniform(150, 230)
            cv2.circle(img, (cx, cy), r, (rim,) * 3, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), int(r * rng.uniform(0.62, 0.8)), (rng.uniform(15, 60),) * 3, -1, cv2.LINE_AA)
            cv2.circle(mask, (cx, cy), r + 2, 255, -1)
            boxes.append((cx - r, cy - r, cx + r, cy + r))
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask, boxes


# ---------- scene composition -------------------------------------------------------------

def background(rng, S):
    c0, c1 = rng.uniform(40, 190, 3), rng.uniform(40, 190, 3)
    t = np.linspace(0, 1, S)[:, None, None]
    bg = (c0 * (1 - t) + c1 * t) * np.ones((S, S, 3))
    for _ in range(int(rng.integers(0, 6))):     # shelves / pillars / floor patches
        x, y = rng.integers(0, S, 2); w, h = rng.integers(20, S // 2, 2)
        cv2.rectangle(bg, (int(x), int(y)), (int(x + w), int(y + h)), tuple(float(v) for v in rng.uniform(30, 220, 3)), -1)
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), rng.uniform(1, 6))
    return bg + rng.normal(0, 4, bg.shape)


def compose(rng, S, kind):
    img, mask, boxes = (plank_stack if kind == 0 else tube_bundle)(rng, S)
    H, W = img.shape[:2]
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    j = lambda a: rng.uniform(-a, a)
    sh, sw = 0.07 * H, 0.07 * W
    dst = src + np.float32([[j(sw), j(sh)], [j(sw), j(sh)], [j(sw), j(sh)], [j(sw), j(sh)]])
    dst += np.float32([(S - W) / 2 + j(S * 0.06), (S - H) / 2 + j(S * 0.06)])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (S, S), flags=cv2.INTER_LINEAR)
    wmask = cv2.warpPerspective(mask, M, (S, S), flags=cv2.INTER_NEAREST) > 0
    out = background(rng, S)
    out[wmask] = warped[wmask]
    pts = np.float32([[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]] for b in boxes]).reshape(-1, 1, 2)
    wp = cv2.perspectiveTransform(pts, M).reshape(-1, 4, 2)
    lab = []
    for q in wp:
        x0, y0 = q.min(0); x1, y1 = q.max(0)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(S, x1), min(S, y1)
        if x1 - x0 > 4 and y1 - y0 > 4:
            lab.append((kind, (x0 + x1) / 2 / S, (y0 + y1) / 2 / S, (x1 - x0) / S, (y1 - y0) / S))
    # photometric degradation: exposure, glare, blur, JPEG
    out = out * rng.uniform(0.6, 1.25) + rng.uniform(-20, 20)
    gl = np.zeros((S, S), np.float32)
    cv2.circle(gl, (int(rng.integers(0, S)), int(rng.integers(0, S))), int(rng.integers(S // 8, S // 3)), 1.0, -1)
    out += cv2.GaussianBlur(gl, (0, 0), S / 12)[..., None] * rng.uniform(0, 70)
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (0, 0), rng.uniform(0.3, 1.6))
    q = int(rng.integers(35, 90))
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    return out, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tube_frac", type=float, default=0.25)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out)
    counts = {}
    for split, n in (("train", a.train), ("val", a.val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            kind = 1 if rng.random() < a.tube_frac else 0
            img, lab = compose(rng, a.size, kind)
            name = f"{split}_{i:04d}"
            cv2.imwrite(str(root / "images" / split / f"{name}.jpg"), img)
            (root / "labels" / split / f"{name}.txt").write_text(
                "\n".join(f"{k} {x:.5f} {y:.5f} {w:.5f} {h:.5f}" for k, x, y, w, h in lab))
            counts[name] = {"class": "tube_end" if kind else "plank_end", "count": len(lab)}
    (root / "counts.json").write_text(json.dumps(counts, indent=1))
    (root / "dataset.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: plank_end\n  1: tube_end\n")
    print("wrote", len(counts), "images to", root)


if __name__ == "__main__":
    main()
