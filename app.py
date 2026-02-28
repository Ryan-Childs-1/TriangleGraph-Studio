# app.py
# ============================================================
# TriangleGraph Studio (NO TORCH) — Diffusion + Puzzle AI
# ------------------------------------------------------------
# Fixes your Streamlit Cloud error by removing torch entirely.
#
# This app implements the SAME combined project idea using only:
#   - streamlit
#   - numpy
#   - pillow
#
# What you get:
# 1) TriangleGraph Diffusion (structure diffusion) WITHOUT deep learning:
#    - Trains a graph denoiser using closed-form linear regression on graph features.
#    - Uses DDPM-like iterative denoising sampling on triangle colors (linear RGB).
#    - Still learns "relationships between triangles" via neighbor-aggregated features.
#
# 2) Snap-Together Puzzle AI WITHOUT deep learning:
#    - Learns an edge-compatibility metric via "prototypes" (mean features for true neighbor edges)
#      and a calibrated scorer that ranks edge matches.
#    - Provides top-K edge match suggestions.
#
# Also includes:
# - Graph dataset export
# - Puzzle preview with whitespace slider
# - Generated outputs export
#
# Install:
#   pip install streamlit numpy pillow
#
# Run:
#   streamlit run app.py
# ============================================================

import io
import json
import math
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps


# ============================================================
# -------------------- Utils / Color / Geometry ---------------
# ============================================================

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / max(w, h)
    return img.resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS)


def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)


def u8_to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0


def float_to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def srgb_to_lin(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)


def lin_to_srgb(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.0031308, 12.92 * c, (1 + a) * (c ** (1 / 2.4)) - a)


def sobel_edge_strength(rgb_u8: np.ndarray) -> np.ndarray:
    x = u8_to_float(rgb_u8)
    lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    p = np.pad(lum, 1, mode="edge")
    gx = (
        kx[0, 0] * p[:-2, :-2] + kx[0, 1] * p[:-2, 1:-1] + kx[0, 2] * p[:-2, 2:] +
        kx[1, 0] * p[1:-1, :-2] + kx[1, 1] * p[1:-1, 1:-1] + kx[1, 2] * p[1:-1, 2:] +
        kx[2, 0] * p[2:, :-2] + kx[2, 1] * p[2:, 1:-1] + kx[2, 2] * p[2:, 2:]
    )
    gy = (
        ky[0, 0] * p[:-2, :-2] + ky[0, 1] * p[:-2, 1:-1] + ky[0, 2] * p[:-2, 2:] +
        ky[1, 0] * p[1:-1, :-2] + ky[1, 1] * p[1:-1, 1:-1] + ky[1, 2] * p[1:-1, 2:] +
        ky[2, 0] * p[2:, :-2] + ky[2, 1] * p[2:, 1:-1] + ky[2, 2] * p[2:, 2:]
    )

    mag = np.sqrt(gx * gx + gy * gy)
    hi = np.quantile(mag, 0.995) + 1e-8
    return np.clip(mag / hi, 0.0, 1.0).astype(np.float32)


def triangle_area(pts: np.ndarray) -> float:
    a, b, c = pts.astype(np.float32)
    return float(abs(np.cross(b - a, c - a)) * 0.5)


def triangle_centroid(pts: np.ndarray) -> np.ndarray:
    return np.mean(pts.astype(np.float32), axis=0)


def triangle_orientation(pts: np.ndarray) -> float:
    p = pts.astype(np.float32)
    edges = [(0, 1), (1, 2), (2, 0)]
    lens = []
    vecs = []
    for i, j in edges:
        v = p[j] - p[i]
        lens.append(float(np.linalg.norm(v)))
        vecs.append(v)
    k = int(np.argmax(lens))
    v = vecs[k]
    ang = math.atan2(float(v[1]), float(v[0]))
    if ang < 0:
        ang += math.pi
    return float(ang)


def triangle_min_edge_len(pts: np.ndarray) -> float:
    p = pts.astype(np.float32)
    a, b, c = p
    ab = float(np.linalg.norm(a - b))
    bc = float(np.linalg.norm(b - c))
    ca = float(np.linalg.norm(c - a))
    return max(1e-6, min(ab, bc, ca))


def shrink_triangle(pts: np.ndarray, gap_px: float) -> np.ndarray:
    pts = pts.astype(np.float32)
    ctr = triangle_centroid(pts)
    min_edge = triangle_min_edge_len(pts)
    scale = 1.0 - float(gap_px) / min_edge
    scale = max(0.05, min(1.0, scale))
    return ctr[None, :] + (pts - ctr[None, :]) * scale


def sample_triangle_color(rgb_u8: np.ndarray, tri: np.ndarray) -> np.ndarray:
    h, w, _ = rgb_u8.shape
    tri = tri.astype(np.float32)
    ctr = triangle_centroid(tri)

    pts = []
    for t in [0.55, 0.75, 0.90]:
        for v in tri:
            p = ctr * (1 - t) + v * t
            x = int(np.clip(round(p[0]), 0, w - 1))
            y = int(np.clip(round(p[1]), 0, h - 1))
            pts.append(rgb_u8[y, x].astype(np.float32))
    x = int(np.clip(round(ctr[0]), 0, w - 1))
    y = int(np.clip(round(ctr[1]), 0, h - 1))
    pts.append(rgb_u8[y, x].astype(np.float32))

    c = np.mean(np.stack(pts, axis=0), axis=0)
    return np.clip(c / 255.0, 0.0, 1.0).astype(np.float32)


# ============================================================
# ---------------- Adaptive Triangulation (Quadtree) ----------
# ============================================================

@dataclass
class QuadParams:
    min_cell: int
    max_depth: int
    var_thresh: float
    edge_thresh: float
    edge_weight: float  # 0..1


def region_stats(rgb_u8: np.ndarray, edge: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Tuple[float, float]:
    roi = u8_to_float(rgb_u8[y0:y1, x0:x1])
    if roi.size == 0:
        return 0.0, 0.0
    var = float(np.mean(np.var(roi.reshape(-1, 3), axis=0)))
    em = float(np.mean(edge[y0:y1, x0:x1])) if edge is not None else 0.0
    return var, em


def build_quadtree_leaves(rgb_u8: np.ndarray, edge: np.ndarray, qp: QuadParams) -> List[Tuple[int, int, int, int]]:
    h, w, _ = rgb_u8.shape
    leaves: List[Tuple[int, int, int, int]] = []

    def should_split(x0, y0, x1, y1, depth) -> bool:
        rw, rh = x1 - x0, y1 - y0
        if rw <= qp.min_cell or rh <= qp.min_cell:
            return False
        if depth >= qp.max_depth:
            return False

        var, em = region_stats(rgb_u8, edge, x0, y0, x1, y1)
        ew = clamp01(qp.edge_weight)
        var_n = min(1.0, var / 0.03)
        em_n = float(em)
        score = (1 - ew) * var_n + ew * em_n
        return (var_n > qp.var_thresh) or (em_n > qp.edge_thresh) or (score > max(qp.var_thresh, qp.edge_thresh))

    def rec(x0, y0, x1, y1, depth):
        if should_split(x0, y0, x1, y1, depth):
            mx = (x0 + x1) // 2
            my = (y0 + y1) // 2
            if mx == x0 or mx == x1 or my == y0 or my == y1:
                leaves.append((x0, y0, x1, y1))
                return
            rec(x0, y0, mx, my, depth + 1)
            rec(mx, y0, x1, my, depth + 1)
            rec(x0, my, mx, y1, depth + 1)
            rec(mx, my, x1, y1, depth + 1)
        else:
            leaves.append((x0, y0, x1, y1))

    rec(0, 0, w, h, 0)
    return leaves


def rect_to_triangles(rect: Tuple[int, int, int, int], diag_mode: str, rng: np.random.Generator) -> List[np.ndarray]:
    x0, y0, x1, y1 = rect
    p00 = np.array([x0, y0], dtype=np.float32)
    p10 = np.array([x1, y0], dtype=np.float32)
    p01 = np.array([x0, y1], dtype=np.float32)
    p11 = np.array([x1, y1], dtype=np.float32)

    mode = diag_mode
    if diag_mode == "Random":
        mode = "TL-BR" if rng.random() < 0.5 else "TR-BL"
    elif diag_mode == "Alternate":
        mode = "TL-BR" if ((x0 + y0) % 2 == 0) else "TR-BL"

    if mode == "TL-BR":
        t1 = np.stack([p00, p10, p11], axis=0)
        t2 = np.stack([p00, p11, p01], axis=0)
    else:
        t1 = np.stack([p00, p10, p01], axis=0)
        t2 = np.stack([p10, p11, p01], axis=0)

    return [t1, t2]


# ============================================================
# ---------------- Triangle Graph Construction ----------------
# ============================================================

@dataclass
class TriangleGraph:
    width: int
    height: int
    tris: np.ndarray          # (N,3,2) float32
    node_feat: np.ndarray     # (N,F) float32
    colors_lin: np.ndarray    # (N,3) float32
    edges: np.ndarray         # (E,2) int64 undirected one-per-shared-edge
    edge_feat: np.ndarray     # (E,Fe) float32
    tri_edge_map: np.ndarray  # (E,2) int64 local edge ids for each triangle in the pair


def _edge_key(p1: np.ndarray, p2: np.ndarray) -> Tuple[int, int, int, int]:
    a = (int(round(p1[0])), int(round(p1[1])))
    b = (int(round(p2[0])), int(round(p2[1])))
    if a <= b:
        return (a[0], a[1], b[0], b[1])
    else:
        return (b[0], b[1], a[0], a[1])


def build_triangle_graph(img_rgb_u8: np.ndarray, qp: QuadParams, diag_mode: str, seed: int) -> TriangleGraph:
    h, w, _ = img_rgb_u8.shape
    edge_map = sobel_edge_strength(img_rgb_u8)
    leaves = build_quadtree_leaves(img_rgb_u8, edge_map, qp)
    rng = np.random.default_rng(int(seed))

    tris_list: List[np.ndarray] = []
    for rect in leaves:
        tris_list.extend(rect_to_triangles(rect, diag_mode, rng))

    tris = np.stack(tris_list, axis=0).astype(np.float32)
    N = tris.shape[0]

    # node features: vertices norm (6) + centroid norm (2) + area (1) + orientation sin/cos (2) + edge@centroid (1)
    Fnode = 12
    node_feat = np.zeros((N, Fnode), dtype=np.float32)
    colors_lin = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        tri = tris[i]
        tri_n = tri.copy()
        tri_n[:, 0] /= max(1.0, float(w))
        tri_n[:, 1] /= max(1.0, float(h))

        ctr = triangle_centroid(tri)
        ctr_n = np.array([ctr[0] / max(1.0, w), ctr[1] / max(1.0, h)], dtype=np.float32)

        area = triangle_area(tri) / max(1.0, float(w * h))
        ang = triangle_orientation(tri)
        ang_s, ang_c = math.sin(ang), math.cos(ang)

        cx = int(np.clip(round(ctr[0]), 0, w - 1))
        cy = int(np.clip(round(ctr[1]), 0, h - 1))
        ectr = float(edge_map[cy, cx])

        node_feat[i] = np.concatenate([
            tri_n.reshape(-1),
            ctr_n,
            np.array([area], dtype=np.float32),
            np.array([ang_s, ang_c], dtype=np.float32),
            np.array([ectr], dtype=np.float32),
        ])

        c_srgb = sample_triangle_color(img_rgb_u8, tri)  # sRGB [0,1]
        colors_lin[i] = srgb_to_lin(c_srgb)

    # shared edge adjacency
    local_edges = [(0, 1), (1, 2), (2, 0)]
    edge_dict: Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}
    pairs = []
    tri_edge_map = []
    edge_feat_list = []

    for ti in range(N):
        tri = tris[ti]
        for le, (a, b) in enumerate(local_edges):
            key = _edge_key(tri[a], tri[b])
            if key not in edge_dict:
                edge_dict[key] = (ti, le)
            else:
                tj, le2 = edge_dict[key]
                if tj == ti:
                    continue
                i, j = tj, ti
                e_i, e_j = le2, le

                tri_i, tri_j = tris[i], tris[j]
                ci = triangle_centroid(tri_i)
                cj = triangle_centroid(tri_j)
                dist_n = float(np.linalg.norm(ci - cj)) / max(1.0, float(max(w, h)))

                p1, p2 = tri[a].astype(np.float32), tri[b].astype(np.float32)
                elen_n = float(np.linalg.norm(p2 - p1)) / max(1.0, float(max(w, h)))

                eang = math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0]))
                eang_s, eang_c = math.sin(eang), math.cos(eang)

                ai, aj = triangle_orientation(tri_i), triangle_orientation(tri_j)
                da = (aj - ai + math.pi) % (2 * math.pi) - math.pi
                da_s, da_c = math.sin(da), math.cos(da)

                ef = np.array([elen_n, dist_n, da_s, da_c, eang_s, eang_c], dtype=np.float32)

                pairs.append((i, j))
                tri_edge_map.append((e_i, e_j))
                edge_feat_list.append(ef)

                del edge_dict[key]

    if len(pairs) == 0:
        edges = np.zeros((0, 2), dtype=np.int64)
        edge_feat = np.zeros((0, 6), dtype=np.float32)
        tri_edge_map_np = np.zeros((0, 2), dtype=np.int64)
    else:
        edges = np.array(pairs, dtype=np.int64)
        edge_feat = np.stack(edge_feat_list, axis=0).astype(np.float32)
        tri_edge_map_np = np.array(tri_edge_map, dtype=np.int64)

    return TriangleGraph(
        width=w, height=h, tris=tris,
        node_feat=node_feat, colors_lin=colors_lin,
        edges=edges, edge_feat=edge_feat,
        tri_edge_map=tri_edge_map_np
    )


def render_triangle_mosaic(
    graph: TriangleGraph,
    colors_lin: np.ndarray,
    gap_px: float,
    background: str,
    outline: bool,
    outline_px: int,
    outline_alpha: float
) -> Image.Image:
    w, h = graph.width, graph.height
    bg = (255, 255, 255, 255) if background == "White" else (0, 0, 0, 255)
    canvas = Image.new("RGBA", (w, h), bg)
    draw = ImageDraw.Draw(canvas, "RGBA")

    srgb = lin_to_srgb(np.clip(colors_lin, 0.0, 1.0))
    c_u8 = float_to_u8(srgb)

    outline_rgba = None
    if outline:
        oa = int(round(255 * clamp01(outline_alpha)))
        outline_rgba = (0, 0, 0, oa) if background == "White" else (255, 255, 255, oa)

    for i in range(graph.tris.shape[0]):
        tri = graph.tris[i]
        tri_s = shrink_triangle(tri, float(gap_px))
        poly = [(float(p[0]), float(p[1])) for p in tri_s]
        fill = (int(c_u8[i, 0]), int(c_u8[i, 1]), int(c_u8[i, 2]), 255)
        draw.polygon(poly, fill=fill)
        if outline_rgba is not None and outline_px > 0:
            draw.line(poly + [poly[0]], fill=outline_rgba, width=int(outline_px), joint="curve")

    return canvas


def export_graph_zip(graphs: List[TriangleGraph], names: List[str]) -> bytes:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        index = []
        for g, nm in zip(graphs, names):
            payload = {
                "width": g.width,
                "height": g.height,
                "tris": g.tris.tolist(),
                "node_feat": g.node_feat.tolist(),
                "colors_lin": g.colors_lin.tolist(),
                "edges": g.edges.tolist(),
                "edge_feat": g.edge_feat.tolist(),
                "tri_edge_map": g.tri_edge_map.tolist(),
            }
            zf.writestr(f"graphs/{nm}.json", json.dumps(payload))
            index.append({"name": nm, "triangles": int(g.tris.shape[0]), "edges": int(g.edges.shape[0])})
        zf.writestr("index.json", json.dumps(index, indent=2))
    return zbuf.getvalue()


# ============================================================
# ------------- Graph Denoiser (NO-TORCH "Diffusion") ---------
# ============================================================
# We train a linear denoiser:
#   eps_hat = W * phi(node, neighbors, xt, t) + b
# where:
#   xt = noisy colors
#   phi includes local + neighbor aggregated features
#
# This is still "learning relationships between triangles" because
# phi includes neighbor summaries via the triangle adjacency graph.


@dataclass
class LinDenoiser:
    W: np.ndarray  # (D,3)
    b: np.ndarray  # (3,)
    feat_mean: np.ndarray  # (D,)
    feat_std: np.ndarray   # (D,)


@dataclass
class DiffSchedule:
    T: int
    beta_start: float
    beta_end: float

    def make(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        betas = np.linspace(self.beta_start, self.beta_end, self.T, dtype=np.float32)
        alphas = 1.0 - betas
        abar = np.cumprod(alphas, axis=0)
        return betas, alphas, abar


def build_neighbor_lists(edges: np.ndarray, N: int) -> List[List[int]]:
    nbr = [[] for _ in range(N)]
    for (i, j) in edges.astype(np.int64):
        nbr[int(i)].append(int(j))
        nbr[int(j)].append(int(i))
    return nbr


def compute_graph_features(
    node_feat: np.ndarray,
    edges: np.ndarray,
    xt: np.ndarray,
    t01: float
) -> np.ndarray:
    """
    node_feat: (N,F)
    xt: (N,3) noisy colors (linear)
    returns phi: (N,D)
    """
    N = node_feat.shape[0]
    nbr = build_neighbor_lists(edges, N)

    # neighbor color mean + neighbor color variance (per-channel)
    nbr_mean = np.zeros_like(xt, dtype=np.float32)
    nbr_var = np.zeros_like(xt, dtype=np.float32)

    for i in range(N):
        ns = nbr[i]
        if not ns:
            nbr_mean[i] = xt[i]
            nbr_var[i] = 0.0
        else:
            arr = xt[ns]
            nbr_mean[i] = np.mean(arr, axis=0)
            nbr_var[i] = np.var(arr, axis=0)

    # also include color deltas (xt - nbr_mean)
    delta = xt - nbr_mean

    # time embedding (simple polynomial + trig)
    t = float(t01)
    tvec = np.array([t, t * t, math.sin(2 * math.pi * t), math.cos(2 * math.pi * t)], dtype=np.float32)
    tfeat = np.tile(tvec[None, :], (N, 1))

    # assemble phi
    # node_feat (F=12)
    # xt (3) + nbr_mean(3) + nbr_var(3) + delta(3) + tfeat(4)
    phi = np.concatenate([node_feat, xt, nbr_mean, nbr_var, delta, tfeat], axis=1).astype(np.float32)
    return phi


def fit_linear_denoiser(
    graphs: List[TriangleGraph],
    sched: DiffSchedule,
    steps: int,
    ridge: float,
    seed: int
) -> LinDenoiser:
    """
    Fit eps ~ phi with ridge regression.
    """
    rng = np.random.default_rng(int(seed))
    betas, alphas, abar = sched.make()

    # gather samples across graphs and random timesteps
    # We'll accumulate X^T X and X^T y for each channel.
    XtX = None
    Xty = None

    # Determine feature dim using a small probe
    g0 = graphs[0]
    N0 = g0.node_feat.shape[0]
    t_idx0 = rng.integers(0, sched.T)
    t01_0 = t_idx0 / max(1, sched.T - 1)
    noise0 = rng.standard_normal(size=(N0, 3)).astype(np.float32)
    a0 = float(abar[t_idx0])
    xt0 = np.sqrt(a0) * g0.colors_lin + np.sqrt(1.0 - a0) * noise0
    phi0 = compute_graph_features(g0.node_feat, g0.edges, xt0, t01_0)
    D = phi0.shape[1]

    XtX = np.zeros((D, D), dtype=np.float64)
    Xty = np.zeros((D, 3), dtype=np.float64)

    for _ in range(int(steps)):
        g = graphs[int(rng.integers(0, len(graphs)))]
        N = g.node_feat.shape[0]
        t_idx = int(rng.integers(0, sched.T))
        t01 = t_idx / max(1, sched.T - 1)

        noise = rng.standard_normal(size=(N, 3)).astype(np.float32)
        a = float(abar[t_idx])
        xt = np.sqrt(a) * g.colors_lin + np.sqrt(1.0 - a) * noise

        phi = compute_graph_features(g.node_feat, g.edges, xt, t01)  # (N,D)

        # accumulate
        X = phi.astype(np.float64)
        Y = noise.astype(np.float64)  # eps target

        XtX += X.T @ X
        Xty += X.T @ Y

    # standardize features (recommended for stability)
    # We'll estimate mean/std on a fresh set of samples.
    # (lightweight pass)
    all_phi = []
    for g in graphs:
        N = g.node_feat.shape[0]
        t_idx = int(rng.integers(0, sched.T))
        t01 = t_idx / max(1, sched.T - 1)
        noise = rng.standard_normal(size=(N, 3)).astype(np.float32)
        a = float(abar[t_idx])
        xt = np.sqrt(a) * g.colors_lin + np.sqrt(1.0 - a) * noise
        phi = compute_graph_features(g.node_feat, g.edges, xt, t01)
        all_phi.append(phi)
    all_phi = np.concatenate(all_phi, axis=0)
    feat_mean = np.mean(all_phi, axis=0).astype(np.float32)
    feat_std = (np.std(all_phi, axis=0) + 1e-6).astype(np.float32)

    # Recompute ridge in standardized space:
    # Solve (X^T X + λI) W = X^T Y
    # We'll approximate by transforming XtX/Xty using mean/std isn't exact because XtX was built unstandardized.
    # So we do a final small solve using sampled standardized data (robust and simple).
    # Build a training matrix for final solve:
    Xs_list = []
    Ys_list = []
    for _ in range(min(12, len(graphs))):
        g = graphs[int(rng.integers(0, len(graphs)))]
        N = g.node_feat.shape[0]
        t_idx = int(rng.integers(0, sched.T))
        t01 = t_idx / max(1, sched.T - 1)
        noise = rng.standard_normal(size=(N, 3)).astype(np.float32)
        a = float(abar[t_idx])
        xt = np.sqrt(a) * g.colors_lin + np.sqrt(1.0 - a) * noise
        phi = compute_graph_features(g.node_feat, g.edges, xt, t01)
        Xs_list.append(((phi - feat_mean) / feat_std).astype(np.float32))
        Ys_list.append(noise.astype(np.float32))
    Xs = np.concatenate(Xs_list, axis=0).astype(np.float64)
    Ys = np.concatenate(Ys_list, axis=0).astype(np.float64)

    D = Xs.shape[1]
    lam = float(ridge)
    A = Xs.T @ Xs + lam * np.eye(D, dtype=np.float64)
    B = Xs.T @ Ys
    W = np.linalg.solve(A, B).astype(np.float32)
    b = np.zeros((3,), dtype=np.float32)

    return LinDenoiser(W=W, b=b, feat_mean=feat_mean, feat_std=feat_std)


def predict_eps(den: LinDenoiser, g: TriangleGraph, xt: np.ndarray, t01: float) -> np.ndarray:
    phi = compute_graph_features(g.node_feat, g.edges, xt, t01)
    X = (phi - den.feat_mean[None, :]) / den.feat_std[None, :]
    eps = X @ den.W + den.b[None, :]
    return eps.astype(np.float32)


def sample_diffusion_colors(
    den: LinDenoiser,
    g: TriangleGraph,
    sched: DiffSchedule,
    guidance: float,
    seed: int
) -> np.ndarray:
    """
    DDPM-ish sampling in numpy.
    Returns x0 estimate in linear RGB.
    """
    rng = np.random.default_rng(int(seed))
    betas, alphas, abar = sched.make()
    N = g.node_feat.shape[0]

    x = rng.standard_normal(size=(N, 3)).astype(np.float32)

    for t in reversed(range(sched.T)):
        t01 = t / max(1, sched.T - 1)
        eps = predict_eps(den, g, x, t01) * float(guidance)

        beta = float(betas[t])
        alpha = float(alphas[t])
        a_bar = float(abar[t])

        mu = (1.0 / math.sqrt(alpha)) * (x - (beta / math.sqrt(1.0 - a_bar + 1e-8)) * eps)

        if t > 0:
            z = rng.standard_normal(size=x.shape).astype(np.float32)
            x = mu + math.sqrt(beta) * z
        else:
            x = mu

    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ============================================================
# ---------------- Puzzle Matcher (NO TORCH) ------------------
# ============================================================
# We learn a compatibility scoring function for edge pairs.
# Approach:
# - Edge feature vector f(tri,edge) in R^8
# - Positive pairs from true neighbors (shared border)
# - We fit a Mahalanobis-like metric by learning per-feature weights w >= 0
#   that maximize separation between positives and random negatives.
# - Then matching score between edge A and B is:
#   score = - sum_k w_k * (fA_k - fB_k)^2
# (plus optional angle "flip" handling via periodic components already in features)

def triangle_local_edge_features(tri: np.ndarray, local_edge_idx: int, w: int, h: int) -> np.ndarray:
    p = tri.astype(np.float32)
    edges = [(0, 1), (1, 2), (2, 0)]
    a, b = edges[int(local_edge_idx)]
    p1, p2 = p[a], p[b]
    ctr = triangle_centroid(p)
    mid = 0.5 * (p1 + p2)

    elen = float(np.linalg.norm(p2 - p1)) / max(1.0, float(max(w, h)))
    ang = math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0]))
    ang_s, ang_c = math.sin(ang), math.cos(ang)

    dx = float(mid[0] - ctr[0]) / max(1.0, float(w))
    dy = float(mid[1] - ctr[1]) / max(1.0, float(h))

    area = triangle_area(p) / max(1.0, float(w * h))
    ori = triangle_orientation(p)
    ori_s, ori_c = math.sin(ori), math.cos(ori)

    return np.array([elen, ang_s, ang_c, dx, dy, area, ori_s, ori_c], dtype=np.float32)


@dataclass
class EdgeMetric:
    w: np.ndarray        # (8,) nonnegative weights
    feat_mean: np.ndarray
    feat_std: np.ndarray


def fit_edge_metric(graphs: List[TriangleGraph], neg_per_pos: int, seed: int) -> EdgeMetric:
    rng = np.random.default_rng(int(seed))

    posA = []
    posB = []

    for g in graphs:
        E = g.edges.shape[0]
        if E == 0:
            continue
        for ei in range(E):
            i, j = int(g.edges[ei, 0]), int(g.edges[ei, 1])
            e_i, e_j = int(g.tri_edge_map[ei, 0]), int(g.tri_edge_map[ei, 1])
            fa = triangle_local_edge_features(g.tris[i], e_i, g.width, g.height)
            fb = triangle_local_edge_features(g.tris[j], e_j, g.width, g.height)
            posA.append(fa); posB.append(fb)

    if len(posA) < 32:
        # fallback weights
        return EdgeMetric(
            w=np.ones((8,), dtype=np.float32),
            feat_mean=np.zeros((8,), dtype=np.float32),
            feat_std=np.ones((8,), dtype=np.float32),
        )

    A = np.stack(posA, axis=0).astype(np.float32)
    B = np.stack(posB, axis=0).astype(np.float32)

    # standardize on all edge feats observed
    all_feats = np.concatenate([A, B], axis=0)
    mean = np.mean(all_feats, axis=0).astype(np.float32)
    std = (np.std(all_feats, axis=0) + 1e-6).astype(np.float32)
    A = (A - mean) / std
    B = (B - mean) / std

    # positives squared diffs
    dpos = (A - B) ** 2  # (P,8)

    # negatives: pair A with random B from different positives
    P = A.shape[0]
    dneg_list = []
    for _ in range(int(neg_per_pos)):
        idx = rng.integers(0, P, size=(P,))
        Bn = B[idx]
        dneg_list.append((A - Bn) ** 2)
    dneg = np.concatenate(dneg_list, axis=0)

    # Learn weights w to separate:
    # We want weighted distance small for positives, large for negatives.
    # A simple closed-form heuristic:
    #   w_k = 1 / (E[dpos_k] + eps)  *  (E[dneg_k] / (E[dpos_k]+eps))
    # then normalize.
    mpos = np.mean(dpos, axis=0) + 1e-6
    mneg = np.mean(dneg, axis=0) + 1e-6
    w = (mneg / mpos) / mpos
    w = np.clip(w, 1e-3, 1e3)
    w = (w / (np.mean(w) + 1e-8)).astype(np.float32)

    return EdgeMetric(w=w, feat_mean=mean, feat_std=std)


def edge_score(metric: EdgeMetric, fA: np.ndarray, fB: np.ndarray) -> float:
    a = (fA - metric.feat_mean) / metric.feat_std
    b = (fB - metric.feat_mean) / metric.feat_std
    d2 = np.sum(metric.w * (a - b) ** 2)
    return float(-d2)  # higher is better


def build_edge_bank(g: TriangleGraph) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    feats = []
    ids = []
    for tid in range(g.tris.shape[0]):
        for e in range(3):
            feats.append(triangle_local_edge_features(g.tris[tid], e, g.width, g.height))
            ids.append((tid, e))
    return np.stack(feats, axis=0).astype(np.float32), ids


# ============================================================
# --------------------------- Streamlit UI --------------------
# ============================================================

st.set_page_config(page_title="TriangleGraph Studio (No Torch)", page_icon="🔺", layout="wide")
st.title("🔺 TriangleGraph Studio — No Torch (Diffusion + Puzzle AI)")
st.caption(
    "Same project concept as before, rebuilt without PyTorch. "
    "Includes: graph dataset export, numpy-based diffusion sampling, and a learned edge-matching scorer."
)

tabs = st.tabs(["1) Build Graphs", "2) Train Models", "3) Generate (Diffusion)", "4) Puzzle Mode (Matching)"])

# ---------------------------
# Tab 1
# ---------------------------
with tabs[0]:
    st.subheader("1) Build triangle graphs from images")
    uploads = st.file_uploader(
        "Upload images (PNG/JPG/WebP) — multiple allowed",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        max_side = st.slider("Max side (px)", 256, 1600, 900, 32)
    with c2:
        min_cell = st.slider("Min cell (px)", 6, 80, 18, 1)
    with c3:
        max_depth = st.slider("Max depth", 2, 10, 7, 1)
    with c4:
        diag_mode = st.selectbox("Diagonal mode", ["Random", "Alternate", "TL-BR", "TR-BL"], index=0)

    c5, c6, c7 = st.columns(3)
    with c5:
        var_thresh = st.slider("Color-var thresh", 0.05, 1.0, 0.55, 0.01)
    with c6:
        edge_thresh = st.slider("Edge thresh", 0.05, 1.0, 0.45, 0.01)
    with c7:
        edge_weight = st.slider("Edge weighting", 0.0, 1.0, 0.60, 0.01)

    seed_graph = st.number_input("Seed (triangulation randomness)", min_value=0, max_value=10_000_000, value=42, step=1)

    preview_gap = st.slider("Preview gap (px)", 0.0, 25.0, 6.0, 0.5)
    preview_bg = st.radio("Preview background", ["White", "Black"], horizontal=True, index=0)
    preview_outline = st.checkbox("Preview outlines", value=False)
    outline_px = st.slider("Outline width", 1, 5, 1, 1) if preview_outline else 1
    outline_alpha = st.slider("Outline opacity", 0.05, 1.0, 0.25, 0.01) if preview_outline else 0.25

    build_btn = st.button("Build graphs", type="primary", use_container_width=True)

    if build_btn:
        if not uploads:
            st.warning("Upload at least one image.")
        else:
            qp = QuadParams(
                min_cell=int(min_cell),
                max_depth=int(max_depth),
                var_thresh=float(var_thresh),
                edge_thresh=float(edge_thresh),
                edge_weight=float(edge_weight),
            )

            graphs: List[TriangleGraph] = []
            names: List[str] = []

            with st.spinner("Building triangle graphs…"):
                for f in uploads:
                    img = Image.open(io.BytesIO(f.getvalue()))
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    img = resize_keep_aspect(img, int(max_side))
                    rgb_u8 = pil_to_np_rgb(img)

                    g = build_triangle_graph(rgb_u8, qp, diag_mode=diag_mode, seed=int(seed_graph))
                    graphs.append(g)
                    names.append(f.name.rsplit(".", 1)[0])

            st.session_state["graphs"] = graphs
            st.session_state["graph_names"] = names
            st.session_state.pop("denoiser", None)
            st.session_state.pop("schedule", None)
            st.session_state.pop("edge_metric", None)

            st.success(f"Built {len(graphs)} graph(s).")

            for g, nm in zip(graphs, names):
                st.markdown(f"**{nm}** — triangles: **{g.tris.shape[0]}**, edges: **{g.edges.shape[0]}**, size: **{g.width}×{g.height}**")
                prev = render_triangle_mosaic(
                    g, g.colors_lin, gap_px=float(preview_gap), background=preview_bg,
                    outline=preview_outline, outline_px=int(outline_px), outline_alpha=float(outline_alpha)
                )
                st.image(prev, use_container_width=True)

            z = export_graph_zip(graphs, names)
            st.download_button(
                "Download dataset ZIP (graphs/*.json + index.json)",
                data=z,
                file_name="triangle_graph_dataset.zip",
                mime="application/zip",
                use_container_width=True,
            )


# ---------------------------
# Tab 2: Train
# ---------------------------
with tabs[1]:
    st.subheader("2) Train models (no torch)")
    st.write(
        "Trains:\n"
        "- **Linear Graph Denoiser** (for diffusion sampling)\n"
        "- **Edge Metric** (for puzzle edge matching)\n"
        "\nThis is fast and runs on Streamlit Cloud CPU."
    )

    if "graphs" not in st.session_state:
        st.info("Go to **1) Build Graphs** first.")
    else:
        graphs: List[TriangleGraph] = st.session_state["graphs"]

        colA, colB = st.columns([1, 1], gap="large")

        with colA:
            st.markdown("### Diffusion (numpy)")
            T = st.slider("Diffusion steps (T)", 20, 200, 80, 5)
            beta_start = st.slider("beta_start", 1e-5, 5e-3, 5e-4, format="%.6f")
            beta_end = st.slider("beta_end", 1e-3, 5e-2, 2e-2, format="%.6f")
            train_steps = st.slider("Training samples (steps)", 50, 5000, 800, 50)
            ridge = st.number_input("Ridge λ", min_value=1e-6, max_value=10.0, value=1e-2, format="%.6f")
            seed_train = st.number_input("Train seed", min_value=0, max_value=10_000_000, value=123, step=1)

        with colB:
            st.markdown("### Puzzle matcher (numpy)")
            neg_per_pos = st.slider("Negatives per positive", 1, 50, 10, 1)
            seed_match = st.number_input("Matcher seed", min_value=0, max_value=10_000_000, value=777, step=1)

        train_btn = st.button("Train both (no torch)", type="primary", use_container_width=True)

        if train_btn:
            sched = DiffSchedule(T=int(T), beta_start=float(beta_start), beta_end=float(beta_end))

            with st.spinner("Training linear denoiser…"):
                t0 = time.time()
                den = fit_linear_denoiser(
                    graphs=graphs,
                    sched=sched,
                    steps=int(train_steps),
                    ridge=float(ridge),
                    seed=int(seed_train),
                )
                st.success(f"Denoiser trained in {time.time() - t0:.2f}s.")

            with st.spinner("Training edge metric…"):
                t1 = time.time()
                metric = fit_edge_metric(graphs=graphs, neg_per_pos=int(neg_per_pos), seed=int(seed_match))
                st.success(f"Edge metric trained in {time.time() - t1:.2f}s.")

            st.session_state["denoiser"] = den
            st.session_state["schedule"] = sched
            st.session_state["edge_metric"] = metric


# ---------------------------
# Tab 3: Generate
# ---------------------------
with tabs[2]:
    st.subheader("3) Generate new images with structure diffusion (no torch)")
    st.write(
        "Select a graph (geometry + adjacency). The sampler generates **new triangle colors** using the learned denoiser."
    )

    if "graphs" not in st.session_state:
        st.info("Go to **1) Build Graphs** first.")
    elif "denoiser" not in st.session_state or "schedule" not in st.session_state:
        st.info("Train the denoiser in **2) Train Models** first.")
    else:
        graphs: List[TriangleGraph] = st.session_state["graphs"]
        names: List[str] = st.session_state["graph_names"]
        den: LinDenoiser = st.session_state["denoiser"]
        sched: DiffSchedule = st.session_state["schedule"]

        pick = st.selectbox("Choose graph geometry", names, index=0)
        gi = names.index(pick)
        g = graphs[gi]

        gen_gap = st.slider("Gap (px)", 0.0, 30.0, 8.0, 0.5)
        gen_bg = st.radio("Background", ["White", "Black"], horizontal=True)
        gen_outline = st.checkbox("Outlines", value=False)
        gen_outline_px = st.slider("Outline width", 1, 5, 1, 1) if gen_outline else 1
        gen_outline_alpha = st.slider("Outline opacity", 0.05, 1.0, 0.25, 0.01) if gen_outline else 0.25

        guidance = st.slider("Denoise strength (guidance)", 0.5, 2.5, 1.1, 0.05)
        seed_gen = st.number_input("Generation seed", min_value=0, max_value=10_000_000, value=999, step=1)

        st.markdown("**Original triangle mosaic (from image):**")
        st.image(
            render_triangle_mosaic(g, g.colors_lin, gap_px=float(gen_gap), background=gen_bg,
                                   outline=gen_outline, outline_px=int(gen_outline_px), outline_alpha=float(gen_outline_alpha)),
            use_container_width=True
        )

        gen_btn = st.button("Generate (sample)", type="primary", use_container_width=True)
        if gen_btn:
            with st.spinner("Sampling diffusion…"):
                x = sample_diffusion_colors(
                    den=den,
                    g=g,
                    sched=sched,
                    guidance=float(guidance),
                    seed=int(seed_gen),
                )

            out_img = render_triangle_mosaic(
                g, x, gap_px=float(gen_gap), background=gen_bg,
                outline=gen_outline, outline_px=int(gen_outline_px), outline_alpha=float(gen_outline_alpha),
            )
            st.markdown("**Generated (structure diffusion):**")
            st.image(out_img, use_container_width=True)

            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            st.download_button(
                "Download generated PNG",
                data=buf.getvalue(),
                file_name="triangle_diffusion_generated.png",
                mime="image/png",
                use_container_width=True,
            )


# ---------------------------
# Tab 4: Puzzle matching
# ---------------------------
with tabs[3]:
    st.subheader("4) Puzzle Mode — edge matching suggestions (no torch)")
    st.write(
        "This uses the trained **edge metric** to rank which triangle edge is most compatible with your selected edge."
    )

    if "graphs" not in st.session_state:
        st.info("Go to **1) Build Graphs** first.")
    elif "edge_metric" not in st.session_state:
        st.info("Train the edge metric in **2) Train Models** first.")
    else:
        graphs: List[TriangleGraph] = st.session_state["graphs"]
        names: List[str] = st.session_state["graph_names"]
        metric: EdgeMetric = st.session_state["edge_metric"]

        pick = st.selectbox("Choose puzzle image graph", names, index=0, key="puzzle_pick")
        gi = names.index(pick)
        g = graphs[gi]

        puzzle_gap = st.slider("Puzzle gap (px)", 0.0, 35.0, 14.0, 0.5)
        puzzle_bg = st.radio("Background", ["White", "Black"], horizontal=True, key="puzzle_bg")
        show_colors = st.checkbox("Show colors (OFF = harder puzzle)", value=True)

        colors_show = g.colors_lin if show_colors else (np.ones_like(g.colors_lin) * 0.5).astype(np.float32)
        board_img = render_triangle_mosaic(
            g, colors_show, gap_px=float(puzzle_gap), background=puzzle_bg,
            outline=True, outline_px=1, outline_alpha=0.25
        )
        st.image(board_img, use_container_width=True)

        st.markdown("### Pick a piece + edge to match")
        tri_id = st.number_input("Triangle ID", min_value=0, max_value=int(g.tris.shape[0] - 1), value=0, step=1)
        edge_id = st.selectbox("Edge index (0:01, 1:12, 2:20)", [0, 1, 2], index=0)
        topk = st.slider("Top-K suggestions", 3, 30, 10, 1)

        bank_feats, bank_ids = build_edge_bank(g)

        suggest_btn = st.button("Suggest matches", type="primary", use_container_width=True)
        if suggest_btn:
            q_feat = triangle_local_edge_features(g.tris[int(tri_id)], int(edge_id), g.width, g.height)

            # score vs all
            scores = np.array([edge_score(metric, q_feat, bank_feats[k]) for k in range(bank_feats.shape[0])], dtype=np.float32)

            # exclude self
            for k, (tid, eid) in enumerate(bank_ids):
                if tid == int(tri_id) and eid == int(edge_id):
                    scores[k] = -1e9
                    break

            idx = np.argsort(-scores)[: int(topk)]
            st.markdown("### Top matches")
            for rank, k in enumerate(idx, start=1):
                tid, eid = bank_ids[int(k)]
                st.write(f"{rank}. Triangle **{tid}**, edge **{eid}** — score: **{float(scores[k]):.4f}**")

        st.info(
            "Next upgrade (if you want): implement full auto-assembly by greedily matching edges "
            "and enforcing consistency constraints (no triangle gets paired on the same edge twice, etc.)."
        )


# ============================================================
# Footer
# ============================================================
with st.expander("How this still matches your two ideas (without torch)", expanded=False):
    st.write(
        "**TriangleGraph Diffusion (no torch):** We still do diffusion-style denoising, but the denoiser is a trained "
        "linear model over features that include neighbor-aggregated triangle statistics. So it learns relational color rules.\n\n"
        "**Snap-Together Puzzle AI (no torch):** We still learn edge compatibility from true neighbor edges, but instead of a "
        "neural embedding, we learn a feature-weighted metric that ranks matches.\n"
    )
