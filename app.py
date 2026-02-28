# app.py
# ============================================================
# TriangleGraph Studio (NO TORCH) — Best Standalone “Trainable AI”
# ============================================================
# What’s new in this version (major upgrade):
# ✅ No PyTorch (Streamlit Cloud-friendly): only streamlit, numpy, pillow
# ✅ A *standalone AI model* you can SAVE/LOAD and iteratively improve:
#    - Graph diffusion denoiser (linear, trained from triangle graphs)
#    - Preference model (human-in-the-loop): user chooses which of two generations is better
#      -> learns a scoring function -> used to pick better generations automatically
# ✅ Training Mode (Pairwise comparisons):
#    - Generate two candidates (diffusion + assembly from bag)
#    - User clicks “A better” or “B better”
#    - Preference model updates via logistic (Bradley–Terry style)
# ✅ Better generation loop:
#    - Can sample many candidates, score them with preference model, and output best
# ✅ Export / Import the full model as JSON (denoiser + schedule + preference weights + stats)
#
# Install:
#   pip install streamlit numpy pillow
#
# Run:
#   streamlit run app.py
#
# ============================================================

import io
import json
import math
import time
import zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps


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
    edge_weight: float


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
    tris: np.ndarray          # (N,3,2)
    node_feat: np.ndarray     # (N,12)
    colors_lin: np.ndarray    # (N,3)
    edges: np.ndarray         # (E,2)
    edge_feat: np.ndarray     # (E,6)
    tri_edge_map: np.ndarray  # (E,2)


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

    node_feat = np.zeros((N, 12), dtype=np.float32)
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
            tri_n.reshape(-1),   # 6
            ctr_n,               # 2
            np.array([area], dtype=np.float32),             # 1
            np.array([ang_s, ang_c], dtype=np.float32),     # 2
            np.array([ectr], dtype=np.float32),             # 1  (edge strength at centroid)
        ])

        c_srgb = sample_triangle_color(img_rgb_u8, tri)
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
        width=w, height=h,
        tris=tris,
        node_feat=node_feat,
        colors_lin=colors_lin,
        edges=edges,
        edge_feat=edge_feat,
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


# ============================================================
# ---------------- Export graph dataset ZIP -------------------
# ============================================================

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
# -------------------- No-Torch Diffusion Model ---------------
# ============================================================

@dataclass
class LinDenoiser:
    W: np.ndarray         # (D,3)
    b: np.ndarray         # (3,)
    feat_mean: np.ndarray # (D,)
    feat_std: np.ndarray  # (D,)
    train_steps_total: int = 0


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


def compute_graph_features(node_feat: np.ndarray, edges: np.ndarray, xt: np.ndarray, t01: float) -> np.ndarray:
    N = node_feat.shape[0]
    nbr = build_neighbor_lists(edges, N)

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

    delta = xt - nbr_mean

    t = float(t01)
    tvec = np.array([t, t * t, math.sin(2 * math.pi * t), math.cos(2 * math.pi * t)], dtype=np.float32)
    tfeat = np.tile(tvec[None, :], (N, 1))

    # (12) + (3 xt) + (3 nbr_mean) + (3 nbr_var) + (3 delta) + (4 time) = 28
    phi = np.concatenate([node_feat, xt, nbr_mean, nbr_var, delta, tfeat], axis=1).astype(np.float32)
    return phi


def _estimate_feat_stats(graphs: List[TriangleGraph], sched: DiffSchedule, seed: int, n_probes: int = 8) -> Tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(int(seed))
    betas, alphas, abar = sched.make()

    phis = []
    D = None
    for _ in range(int(n_probes)):
        g = graphs[int(rng.integers(0, len(graphs)))]
        N = g.node_feat.shape[0]
        t_idx = int(rng.integers(0, sched.T))
        t01 = t_idx / max(1, sched.T - 1)
        noise = rng.standard_normal(size=(N, 3)).astype(np.float32)
        a = float(abar[t_idx])
        xt = np.sqrt(a) * g.colors_lin + np.sqrt(1.0 - a) * noise
        phi = compute_graph_features(g.node_feat, g.edges, xt, t01)
        phis.append(phi)
        D = phi.shape[1]
    X = np.concatenate(phis, axis=0)
    mean = np.mean(X, axis=0).astype(np.float32)
    std = (np.std(X, axis=0) + 1e-6).astype(np.float32)
    return mean, std, int(D)


def fit_linear_denoiser(graphs: List[TriangleGraph], sched: DiffSchedule, steps: int, ridge: float, seed: int) -> LinDenoiser:
    rng = np.random.default_rng(int(seed))
    betas, alphas, abar = sched.make()

    feat_mean, feat_std, D = _estimate_feat_stats(graphs, sched, seed=seed + 11, n_probes=10)
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
        phi = compute_graph_features(g.node_feat, g.edges, xt, t01)

        X = ((phi - feat_mean[None, :]) / feat_std[None, :]).astype(np.float64)
        Y = noise.astype(np.float64)

        XtX += X.T @ X
        Xty += X.T @ Y

    lam = float(ridge)
    A = XtX + lam * np.eye(D, dtype=np.float64)
    W = np.linalg.solve(A, Xty).astype(np.float32)
    b = np.zeros((3,), dtype=np.float32)
    return LinDenoiser(W=W, b=b, feat_mean=feat_mean, feat_std=feat_std, train_steps_total=int(steps))


def continue_train_linear_denoiser(den: LinDenoiser, graphs: List[TriangleGraph], sched: DiffSchedule, steps: int, ridge: float, seed: int) -> LinDenoiser:
    """
    Incremental “continue training”:
    - Uses the existing feature mean/std
    - Re-solves ridge system from fresh samples (simple and stable)
    """
    rng = np.random.default_rng(int(seed))
    betas, alphas, abar = sched.make()

    D = den.W.shape[0]
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
        phi = compute_graph_features(g.node_feat, g.edges, xt, t01)

        X = ((phi - den.feat_mean[None, :]) / den.feat_std[None, :]).astype(np.float64)
        Y = noise.astype(np.float64)

        XtX += X.T @ X
        Xty += X.T @ Y

    lam = float(ridge)
    A = XtX + lam * np.eye(D, dtype=np.float64)
    W_new = np.linalg.solve(A, Xty).astype(np.float32)

    den.W = W_new
    den.train_steps_total += int(steps)
    return den


def predict_eps(den: LinDenoiser, g: TriangleGraph, xt: np.ndarray, t01: float) -> np.ndarray:
    phi = compute_graph_features(g.node_feat, g.edges, xt, t01)
    X = (phi - den.feat_mean[None, :]) / den.feat_std[None, :]
    eps = X @ den.W + den.b[None, :]
    return eps.astype(np.float32)


def sample_diffusion_colors(den: LinDenoiser, g: TriangleGraph, sched: DiffSchedule, guidance: float, seed: int) -> np.ndarray:
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
# -------------------- Assemble-from-Bag ----------------------
# ============================================================

def make_color_bag(N: int, mode: str, training_graphs: List[TriangleGraph], seed: int, jitter: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if mode == "Random (uniform)":
        bag = rng.uniform(0.0, 1.0, size=(N, 3)).astype(np.float32)
    else:
        pool = np.concatenate([g.colors_lin for g in training_graphs], axis=0).astype(np.float32)
        idx = rng.integers(0, pool.shape[0], size=(N,))
        bag = pool[idx].copy()
        if jitter > 0:
            bag = np.clip(bag + rng.normal(0.0, jitter, size=bag.shape).astype(np.float32), 0.0, 1.0)
    return bag


def bucket_indices(colors: np.ndarray, bins: int) -> Dict[Tuple[int, int, int], List[int]]:
    q = np.clip((colors * (bins - 1)).astype(np.int32), 0, bins - 1)
    d: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(colors.shape[0]):
        key = (int(q[i, 0]), int(q[i, 1]), int(q[i, 2]))
        d.setdefault(key, []).append(i)
    return d


def assign_bag_to_target(bag: np.ndarray, target: np.ndarray, bins: int = 20, max_ring: int = 6, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    N = target.shape[0]
    bag_dict = bucket_indices(bag, bins=bins)

    assigned = np.zeros_like(target, dtype=np.float32)
    perm = np.full((N,), -1, dtype=np.int32)

    order = np.arange(N, dtype=np.int32)
    rng.shuffle(order)

    tgt_q = np.clip((target * (bins - 1)).astype(np.int32), 0, bins - 1)

    def pop_best(lst: List[int], tcol: np.ndarray) -> int:
        if len(lst) == 1:
            return lst.pop()
        arr = bag[np.array(lst, dtype=np.int32)]
        d2 = np.sum((arr - tcol[None, :]) ** 2, axis=1)
        k = int(np.argmin(d2))
        idx = int(lst[k])
        lst.pop(k)
        return idx

    remaining = set(range(bag.shape[0]))

    for ii in order:
        tcol = target[ii]
        tq = tgt_q[ii]
        picked = None

        for r in range(0, max_ring + 1):
            candidates = []
            for dr in range(-r, r + 1):
                for dg in range(-r, r + 1):
                    for db in range(-r, r + 1):
                        if abs(dr) + abs(dg) + abs(db) != r:
                            continue
                        rr = int(np.clip(tq[0] + dr, 0, bins - 1))
                        gg = int(np.clip(tq[1] + dg, 0, bins - 1))
                        bb = int(np.clip(tq[2] + db, 0, bins - 1))
                        key = (rr, gg, bb)
                        lst = bag_dict.get(key)
                        if lst and len(lst) > 0:
                            candidates.append(key)

            if candidates:
                best_key = None
                best_d2 = 1e9
                for key in candidates:
                    lst = bag_dict[key]
                    arr = bag[np.array(lst, dtype=np.int32)]
                    d2 = np.sum((arr - tcol[None, :]) ** 2, axis=1)
                    dmin = float(np.min(d2))
                    if dmin < best_d2:
                        best_d2 = dmin
                        best_key = key
                picked = pop_best(bag_dict[best_key], tcol)
                break

        if picked is None:
            if not remaining:
                picked = int(rng.integers(0, bag.shape[0]))
            else:
                sample = rng.choice(np.array(list(remaining), dtype=np.int32), size=min(256, len(remaining)), replace=False)
                arr = bag[sample]
                d2 = np.sum((arr - tcol[None, :]) ** 2, axis=1)
                picked = int(sample[int(np.argmin(d2))])

        perm[ii] = picked
        assigned[ii] = bag[picked]
        if picked in remaining:
            remaining.remove(picked)

    return assigned, perm


def assemble_from_bag(
    den: LinDenoiser,
    sched: DiffSchedule,
    g: TriangleGraph,
    training_graphs: List[TriangleGraph],
    bag_mode: str,
    bag_seed: int,
    bag_jitter: float,
    guidance: float,
    diffusion_seed: int,
    assign_bins: int,
    assign_ring: int,
    assign_seed: int,
) -> Dict[str, np.ndarray]:
    N = g.tris.shape[0]
    bag = make_color_bag(N, bag_mode, training_graphs, seed=bag_seed, jitter=bag_jitter)

    target = sample_diffusion_colors(
        den=den,
        g=g,
        sched=sched,
        guidance=guidance,
        seed=diffusion_seed,
    )

    assembled, perm = assign_bag_to_target(
        bag=bag,
        target=target,
        bins=int(assign_bins),
        max_ring=int(assign_ring),
        seed=int(assign_seed),
    )

    rng = np.random.default_rng(int(bag_seed) + 999)
    perm0 = np.arange(N, dtype=np.int32)
    rng.shuffle(perm0)
    random_place = bag[perm0]

    return {
        "bag": bag,
        "target": target,
        "random_place": random_place,
        "assembled": assembled,
        "perm": perm,
        "perm_random": perm0,
    }


# ============================================================
# ------------- Preference Model (Human-in-the-loop) ----------
# ============================================================
# We train a simple logistic preference model:
#   P(A preferred over B) = sigmoid( w · (feat(A) - feat(B)) )
# - feats are computed from the triangle colors + graph structure
# - w updated by gradient descent on user comparisons
#
# This provides a *standalone learned taste function* that:
# - helps pick better samples among many
# - improves over time with your clicks

@dataclass
class PrefModel:
    w: np.ndarray            # (K,)
    feat_mean: np.ndarray    # (K,)
    feat_std: np.ndarray     # (K,)
    n_comparisons: int = 0
    lr: float = 0.15
    l2: float = 0.01


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def edge_deltas(g: TriangleGraph, colors_lin: np.ndarray) -> np.ndarray:
    if g.edges.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    i = g.edges[:, 0].astype(np.int64)
    j = g.edges[:, 1].astype(np.int64)
    return (colors_lin[i] - colors_lin[j]).astype(np.float32)


def compute_quality_features(g: TriangleGraph, colors_lin: np.ndarray) -> np.ndarray:
    """
    Returns feature vector K describing how “good” an image is.
    K is fixed so pref model can persist.
    Features are designed to capture:
    - smoothness vs contrast (edge-aware)
    - color richness
    - global consistency
    """
    # Edge strengths per node are last node_feat column
    edge_strength = g.node_feat[:, -1].astype(np.float32)  # (N,)

    # Neighbor differences (E,3)
    d = edge_deltas(g, colors_lin)
    if d.shape[0] == 0:
        # fallback: use variance only
        var = np.var(colors_lin, axis=0)
        mean = np.mean(colors_lin, axis=0)
        sat = float(np.mean(np.std(lin_to_srgb(colors_lin), axis=1)))
        return np.array([
            float(np.mean(var)), float(np.max(var)), float(np.mean(mean)), sat,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

    # Per-edge color distance
    dist = np.sqrt(np.sum(d * d, axis=1) + 1e-12)  # (E,)

    # Approximate edge strength per edge = mean of endpoint edge strengths
    i = g.edges[:, 0].astype(np.int64)
    j = g.edges[:, 1].astype(np.int64)
    e_strength = 0.5 * (edge_strength[i] + edge_strength[j])  # (E,)

    # Encourage: high contrast where image edges are strong, smooth where edges weak
    # Compute correlation-like measures:
    s = e_strength
    s_mean = float(np.mean(s))
    dist_mean = float(np.mean(dist))
    cov = float(np.mean((s - s_mean) * (dist - dist_mean)))
    s_var = float(np.mean((s - s_mean) ** 2)) + 1e-8
    d_var = float(np.mean((dist - dist_mean) ** 2)) + 1e-8
    corr = cov / math.sqrt(s_var * d_var)

    # Edge-aware losses:
    # - penalty for high dist when edge_strength is low
    # - penalty for low dist when edge_strength is high
    low = (1.0 - s)
    high = s
    smooth_penalty = float(np.mean(low * dist))            # want small
    edge_miss_penalty = float(np.mean(high * (1.0 / (dist + 1e-3))))  # want small (if dist tiny on strong edge, bad)

    # Global color stats
    var_rgb = np.var(colors_lin, axis=0)
    mean_rgb = np.mean(colors_lin, axis=0)
    max_var = float(np.max(var_rgb))
    mean_var = float(np.mean(var_rgb))
    mean_brightness = float(np.mean(mean_rgb))

    # sRGB “saturation-ish”: std across channels per tri averaged
    sr = lin_to_srgb(np.clip(colors_lin, 0.0, 1.0))
    sat = float(np.mean(np.std(sr, axis=1)))

    # Histogram entropy over coarse bins in RGB (encourage richness but not noise)
    bins = 8
    q = np.clip((sr * (bins - 1)).astype(np.int32), 0, bins - 1)
    key = q[:, 0] * (bins * bins) + q[:, 1] * bins + q[:, 2]
    hist = np.bincount(key, minlength=bins ** 3).astype(np.float32)
    hist = hist / (np.sum(hist) + 1e-8)
    ent = float(-np.sum(hist * np.log(hist + 1e-8)))

    # Total variation like statistic on graph
    tv = float(np.mean(dist))

    # Channel balance: penalize if one channel dominates variance too much
    balance = float(np.std(var_rgb))

    # Feature vector (K=12) — keep stable!
    feats = np.array([
        mean_var,            # 0
        max_var,             # 1
        mean_brightness,     # 2
        sat,                 # 3
        ent,                 # 4
        tv,                  # 5
        corr,                # 6
        smooth_penalty,      # 7 (lower better)
        edge_miss_penalty,   # 8 (lower better)
        balance,             # 9 (lower better)
        float(np.median(dist)),  # 10
        float(np.quantile(dist, 0.90)),  # 11
    ], dtype=np.float32)
    return feats


def init_pref_model(K: int = 12) -> PrefModel:
    w = np.zeros((K,), dtype=np.float32)
    mean = np.zeros((K,), dtype=np.float32)
    std = np.ones((K,), dtype=np.float32)
    return PrefModel(w=w, feat_mean=mean, feat_std=std, n_comparisons=0, lr=0.15, l2=0.01)


def pref_score(pref: PrefModel, feats: np.ndarray) -> float:
    x = (feats - pref.feat_mean) / (pref.feat_std + 1e-6)
    return float(np.dot(pref.w, x))


def update_pref_stats(pref: PrefModel, feats_all: np.ndarray) -> PrefModel:
    if feats_all.shape[0] < 4:
        return pref
    pref.feat_mean = np.mean(feats_all, axis=0).astype(np.float32)
    pref.feat_std = (np.std(feats_all, axis=0) + 1e-6).astype(np.float32)
    return pref


def train_pref_model(pref: PrefModel, pairs: List[Dict], steps: int = 200) -> PrefModel:
    """
    pairs: list of dicts with:
      - fa: feature vector
      - fb: feature vector
      - y: 1 if A preferred else 0
    """
    if len(pairs) < 2:
        return pref

    A = np.stack([p["fa"] for p in pairs], axis=0).astype(np.float32)
    B = np.stack([p["fb"] for p in pairs], axis=0).astype(np.float32)
    y = np.array([p["y"] for p in pairs], dtype=np.float32)  # (N,)

    # update normalization stats
    all_feats = np.concatenate([A, B], axis=0)
    pref = update_pref_stats(pref, all_feats)

    # standardized diffs
    Az = (A - pref.feat_mean[None, :]) / pref.feat_std[None, :]
    Bz = (B - pref.feat_mean[None, :]) / pref.feat_std[None, :]
    X = Az - Bz  # (N,K)

    w = pref.w.astype(np.float32)

    # gradient descent
    lr = float(pref.lr)
    l2 = float(pref.l2)
    n = X.shape[0]
    for _ in range(int(steps)):
        logits = X @ w
        p = sigmoid(logits)
        # gradient of logloss: X^T (p - y) / n + l2*w
        grad = (X.T @ (p - y)) / max(1.0, float(n)) + l2 * w
        w = w - lr * grad.astype(np.float32)

    pref.w = w.astype(np.float32)
    pref.n_comparisons = len(pairs)
    return pref


# ============================================================
# -------------------- Model Save / Load ----------------------
# ============================================================

@dataclass
class StandaloneModel:
    # Denoiser
    denoiser: Optional[LinDenoiser]
    schedule: Optional[DiffSchedule]
    # Preference
    pref: PrefModel
    pref_pairs: List[Dict]  # stored comparisons
    # Metadata
    version: str = "trianglegraph-standalone-v1"


def model_to_json_bytes(sm: StandaloneModel) -> bytes:
    payload = {
        "version": sm.version,
        "schedule": None if sm.schedule is None else asdict(sm.schedule),
        "denoiser": None if sm.denoiser is None else {
            "W": sm.denoiser.W.tolist(),
            "b": sm.denoiser.b.tolist(),
            "feat_mean": sm.denoiser.feat_mean.tolist(),
            "feat_std": sm.denoiser.feat_std.tolist(),
            "train_steps_total": int(sm.denoiser.train_steps_total),
        },
        "pref": {
            "w": sm.pref.w.tolist(),
            "feat_mean": sm.pref.feat_mean.tolist(),
            "feat_std": sm.pref.feat_std.tolist(),
            "n_comparisons": int(sm.pref.n_comparisons),
            "lr": float(sm.pref.lr),
            "l2": float(sm.pref.l2),
        },
        "pref_pairs": sm.pref_pairs,
    }
    return json.dumps(payload).encode("utf-8")


def json_bytes_to_model(b: bytes) -> StandaloneModel:
    d = json.loads(b.decode("utf-8"))
    sched = None
    den = None

    if d.get("schedule") is not None:
        s = d["schedule"]
        sched = DiffSchedule(T=int(s["T"]), beta_start=float(s["beta_start"]), beta_end=float(s["beta_end"]))

    if d.get("denoiser") is not None:
        dn = d["denoiser"]
        den = LinDenoiser(
            W=np.array(dn["W"], dtype=np.float32),
            b=np.array(dn["b"], dtype=np.float32),
            feat_mean=np.array(dn["feat_mean"], dtype=np.float32),
            feat_std=np.array(dn["feat_std"], dtype=np.float32),
            train_steps_total=int(dn.get("train_steps_total", 0)),
        )

    pr = d.get("pref", {})
    pref = PrefModel(
        w=np.array(pr.get("w", [0]*12), dtype=np.float32),
        feat_mean=np.array(pr.get("feat_mean", [0]*12), dtype=np.float32),
        feat_std=np.array(pr.get("feat_std", [1]*12), dtype=np.float32),
        n_comparisons=int(pr.get("n_comparisons", 0)),
        lr=float(pr.get("lr", 0.15)),
        l2=float(pr.get("l2", 0.01)),
    )

    pairs = d.get("pref_pairs", [])
    return StandaloneModel(denoiser=den, schedule=sched, pref=pref, pref_pairs=pairs, version=d.get("version", "trianglegraph-standalone-v1"))


# ============================================================
# -------------------- Candidate Generation -------------------
# ============================================================

def generate_candidate(
    den: LinDenoiser,
    sched: DiffSchedule,
    g: TriangleGraph,
    graphs_train: List[TriangleGraph],
    bag_mode: str,
    bag_seed: int,
    bag_jitter: float,
    guidance: float,
    diffusion_seed: int,
    assign_bins: int,
    assign_ring: int,
    assign_seed: int,
) -> Dict:
    out = assemble_from_bag(
        den=den,
        sched=sched,
        g=g,
        training_graphs=graphs_train,
        bag_mode=bag_mode,
        bag_seed=bag_seed,
        bag_jitter=bag_jitter,
        guidance=guidance,
        diffusion_seed=diffusion_seed,
        assign_bins=assign_bins,
        assign_ring=assign_ring,
        assign_seed=assign_seed,
    )
    feats = compute_quality_features(g, out["assembled"])
    return {
        "params": {
            "bag_mode": bag_mode,
            "bag_seed": int(bag_seed),
            "bag_jitter": float(bag_jitter),
            "guidance": float(guidance),
            "diffusion_seed": int(diffusion_seed),
            "assign_bins": int(assign_bins),
            "assign_ring": int(assign_ring),
            "assign_seed": int(assign_seed),
        },
        "out": out,
        "feats": feats,
    }


def pick_best_of_many(pref: PrefModel, candidates: List[Dict]) -> Tuple[int, List[float]]:
    scores = []
    for c in candidates:
        scores.append(pref_score(pref, c["feats"]))
    best_idx = int(np.argmax(np.array(scores)))
    return best_idx, scores


# ============================================================
# --------------------------- Streamlit UI --------------------
# ============================================================

st.set_page_config(page_title="TriangleGraph Studio — Trainable AI (No Torch)", page_icon="🔺", layout="wide")
st.title("🔺 TriangleGraph Studio — Trainable Standalone AI (No Torch)")
st.caption(
    "You can: build graphs → train a diffusion denoiser → generate & assemble from a bag → "
    "**train via human preference** (choose between two generations) → save/load your model."
)

# Initialize session model holder
if "standalone" not in st.session_state:
    st.session_state["standalone"] = StandaloneModel(
        denoiser=None,
        schedule=None,
        pref=init_pref_model(K=12),
        pref_pairs=[],
        version="trianglegraph-standalone-v1"
    )

tabs = st.tabs([
    "1) Build Graphs",
    "2) Train Denoiser",
    "3) Generate & Assemble",
    "4) Preference Training (A/B)",
    "5) Save / Load Model",
])


# ============================================================
# Tab 1: Build Graphs
# ============================================================
with tabs[0]:
    st.subheader("1) Build triangle graphs from images")
    uploads = st.file_uploader("Upload images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        max_side = st.slider("Max side (px)", 256, 16000, 900, 32) # made image 10x size
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

    seed_graph = st.number_input("Seed (triangulation)", 0, 10_000_000, 42, 1)
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
                "Download graph dataset ZIP",
                data=z,
                file_name="triangle_graph_dataset.zip",
                mime="application/zip",
                use_container_width=True,
            )


# ============================================================
# Tab 2: Train Denoiser (and continue training)
# ============================================================
with tabs[1]:
    st.subheader("2) Train diffusion denoiser (standalone)")
    sm: StandaloneModel = st.session_state["standalone"]

    if "graphs" not in st.session_state:
        st.info("Go to **1) Build Graphs** first.")
    else:
        graphs: List[TriangleGraph] = st.session_state["graphs"]

        cA, cB = st.columns([1, 1], gap="large")
        with cA:
            st.markdown("### Schedule")
            T = st.slider("T (steps)", 20, 200, 80, 5)
            beta_start = st.slider("beta_start", 1e-5, 5e-3, 5e-4, format="%.6f")
            beta_end = st.slider("beta_end", 1e-3, 5e-2, 2e-2, format="%.6f")
        with cB:
            st.markdown("### Training")
            train_steps = st.slider("Training samples", 50, 12000, 1800, 50)
            ridge = st.number_input("Ridge λ", min_value=1e-6, max_value=10.0, value=1e-2, format="%.6f")
            seed_train = st.number_input("Train seed", 0, 10_000_000, 123, 1)

        col1, col2 = st.columns(2)
        train_btn = col1.button("Train from scratch", type="primary", use_container_width=True)
        cont_btn = col2.button("Continue training (refine)", use_container_width=True)

        if train_btn:
            sched = DiffSchedule(T=int(T), beta_start=float(beta_start), beta_end=float(beta_end))
            with st.spinner("Training denoiser…"):
                t0 = time.time()
                den = fit_linear_denoiser(graphs, sched, steps=int(train_steps), ridge=float(ridge), seed=int(seed_train))
            sm.denoiser = den
            sm.schedule = sched
            st.success(f"Trained denoiser in {time.time() - t0:.2f}s. Steps total: {den.train_steps_total}")

        if cont_btn:
            if sm.denoiser is None or sm.schedule is None:
                st.warning("Train from scratch first.")
            else:
                # If user changed schedule, warn and keep existing schedule stable (best practice)
                if sm.schedule.T != int(T) or abs(sm.schedule.beta_start - float(beta_start)) > 1e-12 or abs(sm.schedule.beta_end - float(beta_end)) > 1e-12:
                    st.warning("Continue-training keeps the existing schedule for stability. (Re-train from scratch to change schedule.)")

                with st.spinner("Continuing training…"):
                    t0 = time.time()
                    sm.denoiser = continue_train_linear_denoiser(
                        sm.denoiser, graphs, sm.schedule, steps=int(train_steps), ridge=float(ridge), seed=int(seed_train) + 999
                    )
                st.success(f"Refined denoiser in {time.time() - t0:.2f}s. Steps total: {sm.denoiser.train_steps_total}")

        if sm.denoiser is not None and sm.schedule is not None:
            st.info(f"Current denoiser trained steps: **{sm.denoiser.train_steps_total}** • Schedule T={sm.schedule.T}")


# ============================================================
# Tab 3: Generate & Assemble (with best-of-N using preference)
# ============================================================
with tabs[2]:
    st.subheader("3) Generate & Assemble (best-of-N optional)")
    sm: StandaloneModel = st.session_state["standalone"]

    if "graphs" not in st.session_state:
        st.info("Go to **1) Build Graphs** first.")
    elif sm.denoiser is None or sm.schedule is None:
        st.info("Train the denoiser in **2) Train Denoiser** first.")
    else:
        graphs: List[TriangleGraph] = st.session_state["graphs"]
        names: List[str] = st.session_state["graph_names"]

        pick = st.selectbox("Choose geometry graph", names, index=0)
        gi = names.index(pick)
        g = graphs[gi]

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("### Bag + assembly settings")
            bag_mode = st.selectbox("Bag mode", ["Sample from training colors", "Random (uniform)"], index=0)
            bag_jitter = st.slider("Bag jitter (sampled mode)", 0.0, 0.25, 0.05, 0.01)
            assign_bins = st.slider("Assignment bins", 8, 48, 20, 1)
            assign_ring = st.slider("Assignment search radius", 1, 12, 6, 1)

        with right:
            st.markdown("### Diffusion settings")
            guidance = st.slider("Guidance (denoise strength)", 0.5, 2.5, 1.15, 0.05)
            base_seed = st.number_input("Base seed", 0, 10_000_000, 2026, 1)
            best_of = st.slider("Best-of-N candidates", 1, 24, 6, 1, help="If >1, we generate N and pick best by learned preference score.")

        render_gap = st.slider("Render gap (px)", 0.0, 30.0, 10.0, 0.5)
        render_bg = st.radio("Background", ["White", "Black"], horizontal=True, index=0)
        render_outline = st.checkbox("Outlines", value=False)
        opx = st.slider("Outline width", 1, 5, 1, 1) if render_outline else 1
        oal = st.slider("Outline opacity", 0.05, 1.0, 0.25, 0.01) if render_outline else 0.25

        gen_btn = st.button("Generate", type="primary", use_container_width=True)

        st.markdown("**Original (from image):**")
        st.image(render_triangle_mosaic(g, g.colors_lin, render_gap, render_bg, False, 1, 0.2), use_container_width=True)

        if gen_btn:
            with st.spinner("Generating candidates…"):
                candidates = []
                for k in range(int(best_of)):
                    # diversify seeds + small param noise
                    bag_seed = int(base_seed) + 1000 * k + 7
                    diffusion_seed = int(base_seed) + 1000 * k + 19
                    assign_seed = int(base_seed) + 1000 * k + 33

                    # tiny exploration around guidance & jitter if best_of>1
                    if best_of > 1:
                        rng = np.random.default_rng(int(base_seed) + k)
                        gk = float(np.clip(guidance + rng.normal(0.0, 0.06), 0.5, 2.5))
                        jk = float(np.clip(bag_jitter + rng.normal(0.0, 0.015), 0.0, 0.25))
                    else:
                        gk = float(guidance)
                        jk = float(bag_jitter)

                    cand = generate_candidate(
                        den=sm.denoiser,
                        sched=sm.schedule,
                        g=g,
                        graphs_train=graphs,
                        bag_mode=bag_mode,
                        bag_seed=bag_seed,
                        bag_jitter=jk,
                        guidance=gk,
                        diffusion_seed=diffusion_seed,
                        assign_bins=int(assign_bins),
                        assign_ring=int(assign_ring),
                        assign_seed=assign_seed,
                    )
                    candidates.append(cand)

                best_idx, scores = pick_best_of_many(sm.pref, candidates)
                best = candidates[best_idx]
                st.session_state["last_generation"] = {"pick": pick, "candidate": best, "scores": scores}

            # Render results
            assembled = best["out"]["assembled"]
            random_place = best["out"]["random_place"]
            target = best["out"]["target"]
            feats = best["feats"]
            sc = pref_score(sm.pref, feats)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("### Random placement (baseline)")
                st.image(render_triangle_mosaic(g, random_place, render_gap, render_bg, render_outline, opx, oal), use_container_width=True)
            with c2:
                st.markdown(f"### Best assembled (preference score: {sc:.3f})")
                st.image(render_triangle_mosaic(g, assembled, render_gap, render_bg, render_outline, opx, oal), use_container_width=True)

            with st.expander("Show unconstrained diffusion target", expanded=False):
                st.image(render_triangle_mosaic(g, target, render_gap, render_bg, True, 1, 0.25), use_container_width=True)

            with st.expander("Show quality features (debug)", expanded=False):
                st.write(feats)

            # download
            img = render_triangle_mosaic(g, assembled, render_gap, render_bg, render_outline, opx, oal)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button("Download PNG", data=buf.getvalue(), file_name="triangle_best.png", mime="image/png", use_container_width=True)


# ============================================================
# Tab 4: Preference Training (A/B choices)
# ============================================================
with tabs[3]:
    st.subheader("4) Preference Training Mode (A/B)")
    st.write(
        "This is the *human-in-the-loop* trainer.\n\n"
        "Each round:\n"
        "1) The AI generates two candidates (A and B).\n"
        "2) You choose which is better.\n"
        "3) The preference model updates and becomes better at selecting good generations.\n"
    )

    sm: StandaloneModel = st.session_state["standalone"]

    if "graphs" not in st.session_state:
        st.info("Go to **1) Build Graphs** first.")
    elif sm.denoiser is None or sm.schedule is None:
        st.info("Train the denoiser in **2) Train Denoiser** first.")
    else:
        graphs: List[TriangleGraph] = st.session_state["graphs"]
        names: List[str] = st.session_state["graph_names"]

        pick = st.selectbox("Choose geometry graph (slots)", names, index=0, key="ab_pick")
        gi = names.index(pick)
        g = graphs[gi]

        colA, colB = st.columns([1, 1], gap="large")
        with colA:
            st.markdown("### Candidate settings")
            bag_mode = st.selectbox("Bag mode", ["Sample from training colors", "Random (uniform)"], index=0, key="ab_bagmode")
            bag_jitter = st.slider("Bag jitter", 0.0, 0.25, 0.06, 0.01, key="ab_jit")
            assign_bins = st.slider("Assignment bins", 8, 48, 20, 1, key="ab_bins")
            assign_ring = st.slider("Assignment search radius", 1, 12, 6, 1, key="ab_ring")
        with colB:
            st.markdown("### Exploration")
            base_seed = st.number_input("Base seed", 0, 10_000_000, 9001, 1, key="ab_seed")
            guidance_center = st.slider("Guidance center", 0.5, 2.5, 1.15, 0.05, key="ab_guid")
            guidance_spread = st.slider("Guidance spread", 0.0, 0.40, 0.10, 0.01, key="ab_gspread")
            jitter_spread = st.slider("Jitter spread", 0.0, 0.10, 0.02, 0.01, key="ab_jspread")

        render_gap = st.slider("Render gap (px)", 0.0, 30.0, 10.0, 0.5, key="ab_gap")
        render_bg = st.radio("Background", ["White", "Black"], horizontal=True, index=0, key="ab_bg")
        outline = st.checkbox("Outlines", value=False, key="ab_out")
        opx = st.slider("Outline width", 1, 5, 1, 1, key="ab_opx") if outline else 1
        oal = st.slider("Outline opacity", 0.05, 1.0, 0.25, 0.01, key="ab_oal") if outline else 0.25

        rounds_train = st.slider("Gradient steps per update", 50, 600, 200, 25, help="More steps = stronger preference update per click.")
        sm.pref.lr = st.slider("Pref learning rate", 0.02, 0.50, float(sm.pref.lr), 0.01)
        sm.pref.l2 = st.slider("Pref L2", 0.0, 0.20, float(sm.pref.l2), 0.005)

        gen_pair_btn = st.button("Generate A/B pair", type="primary", use_container_width=True)

        if gen_pair_btn:
            rng = np.random.default_rng(int(base_seed))
            gA = float(np.clip(guidance_center + rng.normal(0.0, guidance_spread), 0.5, 2.5))
            gB = float(np.clip(guidance_center + rng.normal(0.0, guidance_spread), 0.5, 2.5))
            jA = float(np.clip(bag_jitter + rng.normal(0.0, jitter_spread), 0.0, 0.25))
            jB = float(np.clip(bag_jitter + rng.normal(0.0, jitter_spread), 0.0, 0.25))

            A = generate_candidate(
                den=sm.denoiser, sched=sm.schedule, g=g, graphs_train=graphs,
                bag_mode=bag_mode, bag_seed=int(base_seed) + 7, bag_jitter=jA,
                guidance=gA, diffusion_seed=int(base_seed) + 17,
                assign_bins=int(assign_bins), assign_ring=int(assign_ring), assign_seed=int(base_seed) + 27,
            )
            B = generate_candidate(
                den=sm.denoiser, sched=sm.schedule, g=g, graphs_train=graphs,
                bag_mode=bag_mode, bag_seed=int(base_seed) + 1007, bag_jitter=jB,
                guidance=gB, diffusion_seed=int(base_seed) + 1017,
                assign_bins=int(assign_bins), assign_ring=int(assign_ring), assign_seed=int(base_seed) + 1027,
            )

            st.session_state["ab_pair"] = {"pick": pick, "A": A, "B": B}

        if "ab_pair" in st.session_state and st.session_state["ab_pair"]["pick"] == pick:
            pair = st.session_state["ab_pair"]
            A = pair["A"]; B = pair["B"]

            a_img = render_triangle_mosaic(g, A["out"]["assembled"], render_gap, render_bg, outline, opx, oal)
            b_img = render_triangle_mosaic(g, B["out"]["assembled"], render_gap, render_bg, outline, opx, oal)

            a_sc = pref_score(sm.pref, A["feats"])
            b_sc = pref_score(sm.pref, B["feats"])

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown(f"## A (score {a_sc:.3f})")
                st.image(a_img, use_container_width=True)
                st.caption(f"params: guidance={A['params']['guidance']:.3f}, jitter={A['params']['bag_jitter']:.3f}")
            with c2:
                st.markdown(f"## B (score {b_sc:.3f})")
                st.image(b_img, use_container_width=True)
                st.caption(f"params: guidance={B['params']['guidance']:.3f}, jitter={B['params']['bag_jitter']:.3f}")

            bA, bB, bSkip = st.columns([1, 1, 1])
            chooseA = bA.button("✅ A is better", use_container_width=True)
            chooseB = bB.button("✅ B is better", use_container_width=True)
            skip = bSkip.button("↩ Skip", use_container_width=True)

            if chooseA or chooseB:
                # store comparison (features only)
                entry = {
                    "fa": A["feats"].tolist(),
                    "fb": B["feats"].tolist(),
                    "y": 1.0 if chooseA else 0.0,
                    "meta": {"pick": pick, "time": time.time()}
                }
                sm.pref_pairs.append(entry)

                # train preference model
                with st.spinner("Updating preference model…"):
                    sm.pref = train_pref_model(sm.pref, sm.pref_pairs, steps=int(rounds_train))

                st.success(f"Preference updated. Comparisons stored: {len(sm.pref_pairs)}")
                # Generate a fresh pair next if user wants
                st.session_state.pop("ab_pair", None)

            if skip:
                st.session_state.pop("ab_pair", None)

        st.divider()
        st.markdown("### Preference status")
        st.write(f"Comparisons: **{len(sm.pref_pairs)}**")
        st.write("Current preference weights (w):")
        st.write(sm.pref.w)


# ============================================================
# Tab 5: Save / Load the full standalone model
# ============================================================
with tabs[4]:
    st.subheader("5) Save / Load Model")
    sm: StandaloneModel = st.session_state["standalone"]

    st.write(
        "This exports/imports your standalone AI model:\n"
        "- Denoiser weights\n"
        "- Diffusion schedule\n"
        "- Preference weights + comparison dataset\n"
    )

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("### Save model")
        name = st.text_input("Model filename", value="trianglegraph_model.json")
        b = model_to_json_bytes(sm)
        st.download_button("Download model JSON", data=b, file_name=name, mime="application/json", use_container_width=True)
        st.caption("Tip: Keep versions as you train more comparisons.")

    with c2:
        st.markdown("### Load model")
        up = st.file_uploader("Upload a saved model JSON", type=["json"])
        if up is not None:
            try:
                sm2 = json_bytes_to_model(up.getvalue())
                st.session_state["standalone"] = sm2
                st.success("Loaded model successfully.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    st.divider()
    st.markdown("### Model summary")
    st.write(f"Version: **{sm.version}**")
    st.write(f"Denoiser trained: **{sm.denoiser is not None}**")
    if sm.denoiser is not None:
        st.write(f"Denoiser steps total: **{sm.denoiser.train_steps_total}**")
    st.write(f"Schedule set: **{sm.schedule is not None}**")
    if sm.schedule is not None:
        st.write(f"Schedule T={sm.schedule.T}, beta_start={sm.schedule.beta_start}, beta_end={sm.schedule.beta_end}")
    st.write(f"Preference comparisons: **{len(sm.pref_pairs)}**")


# ============================================================
# Small footer: practical next steps
# ============================================================
with st.expander("Recommended best-practice workflow", expanded=False):
    st.write(
        "1) Build graphs from 3–10 images (moderate triangle count).\n"
        "2) Train denoiser (2–6k steps), then generate a few times.\n"
        "3) Go to Preference Training and do ~30–100 A/B selections.\n"
        "4) Use Generate & Assemble with Best-of-N (6–24) — preference model picks better outputs.\n"
        "5) Save model JSON frequently as checkpoints.\n"
    )
