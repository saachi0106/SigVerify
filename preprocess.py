"""Shared preprocessing used by training AND the web app (so they never disagree)."""
import cv2
import numpy as np

H, W = 128, 256  # network input size


def to_gray(data) -> np.ndarray:
    """data: file path, raw bytes, or ndarray -> uint8 grayscale."""
    if isinstance(data, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    elif isinstance(data, np.ndarray):
        img = data if data.ndim == 2 else cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(str(data), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image")
    return img


def preprocess(data) -> np.ndarray:
    """Gray -> Otsu ink mask (ink=255) -> denoise -> tight crop -> aspect-preserving
    resize -> centre on H x W canvas. Returns uint8 (H, W)."""
    img = to_gray(data)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # drop tiny specks (scan noise) so they don't distort the bounding box
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            keep[lab == i] = 255
    ink = keep if keep.any() else ink
    ys, xs = np.where(ink > 0)
    if len(xs):
        ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    s = min(H * 0.92 / h, W * 0.92 / w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    ink = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W), np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = ink
    return canvas
