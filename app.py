# app.py
# ============================================================
# TriangleGraph Studio (Streamlit)
# ------------------------------------------------------------
# Combines two novel ideas using ONLY Streamlit-compatible pkgs:
#   1) TriangleGraph Diffusion (graph-based denoising diffusion on triangle colors)
#   2) Snap-Together Puzzle AI (edge-compatibility embedding + neighbor suggestion)
#
# Dependencies (Streamlit-friendly):
#   pip install streamlit numpy pillow torch
#
# Run:
#   streamlit run app.py
#
# Notes:
# - This is intentionally self-contained (no torch-geometric, no opencv, no sklearn).
# - It can train quickly on 1–N uploaded images (small models).
# - Diffusion is "structure diffusion": it generates triangle colors given triangle geometry+adjacency.
# - Puzzle AI learns which edges belong together (contrastive learning), then suggests matches.
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

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# -------------------- Geometry + Image Utils -----------------
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


def float_to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def u8_to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0


def srgb_to_lin(c: np.ndarray) -> np.ndarray:
    # c in [0,1]
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)


def lin_to_srgb(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.0031308, 12.92 * c, (1 + a) * (c ** (1 / 2.4)) - a)


def sobel_edge_strength(rgb_u8: np.ndarray) -> np.ndarray:
    """Edge strength map in [0,1]."""
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
    # pts (3,2)
    a, b, c = pts.astype(np.float32)
    return float(abs(np.cross(b - a, c - a)) * 0.5)


def triangle_centroid(pts: np.ndarray) -> np.ndarray:
    return np.mean(pts.astype(np.float32), axis=0)


def triangle_orientation(pts: np.ndarray) -> float:
    # an orientation "angle" based on the longest edge direction (0..pi)
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
    ang = math.atan2(float(v[1]), float(v[0]))  # -pi..pi
    if ang < 0:
        ang += math.pi  # fold to 0..pi
    return float(ang)


def triangle_min_edge_len(pts: np.ndarray) -> float:
    p = pts.astype(np.float32)
    a, b, c = p
    ab = float(np.linalg.norm(a - b))
    bc = float(np.linalg.norm(b - c))
    ca = float(np.linalg.norm(c - a))
    return max(1e-6, min(ab, bc, ca))


def shrink_triangle(pts: np.ndarray, gap_px: float) -> np.ndarray:
    """
    Shrink toward centroid to create whitespace between puzzle pieces.
    """
    pts = pts.astype(np.float32)
    ctr = triangle_centroid(pts)
    min_edge = triangle_min_edge_len(pts)
    scale = 1.0 - float(gap_px) / min_edge
    scale = max(0.05, min(1.0, scale))
    return ctr[None, :] + (pts - ctr[None, :]) * scale


def sample_triangle_color(rgb_u8: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """
    Sample a color from the original image by averaging a few points.
    Returns sRGB float in [0,1].
    """
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
        var_n = min(1.0, var / 0.03)  # normalize to ~[0,1]
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
        # deterministic parity trick
        mode = "TL-BR" if ((x0 + y0) % 2 == 0) else "TR-BL"

    if mode == "TL-BR":
        t1 = np.stack([p00, p10, p11], axis=0)
        t2 = np.stack([p00, p11, p01], axis=0)
    else:  # "TR-BL"
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
    tris: np.ndarray              # (N,3,2) float32 pixel coords
    node_feat: np.ndarray         # (N,F) float32
    colors_lin: np.ndarray        # (N,3) float32 linear RGB [0,1]
    edges: np.ndarray             # (E,2) int64 node indices (undirected, one per shared edge)
    edge_feat: np.ndarray         # (E,Fe) float32
    tri_edge_map: np.ndarray      # (E,2) int64: for each graph edge, which local edge index (0..2) in each triangle


def _edge_key(p1: np.ndarray, p2: np.ndarray) -> Tuple[int, int, int, int]:
    # Use integer endpoints to create stable shared-edge keys
    a = (int(round(p1[0])), int(round(p1[1])))
    b = (int(round(p2[0])), int(round(p2[1])))
    if a <= b:
        return (a[0], a[1], b[0], b[1])
    else:
        return (b[0], b[1], a[0], a[1])


def build_triangle_graph(
    img_rgb_u8: np.ndarray,
    qp: QuadParams,
    diag_mode: str,
    seed: int
) -> TriangleGraph:
    h, w, _ = img_rgb_u8.shape
    edge_map = sobel_edge_strength(img_rgb_u8)

    leaves = build_quadtree_leaves(img_rgb_u8, edge_map, qp)
    rng = np.random.default_rng(int(seed))

    tris_list: List[np.ndarray] = []
    for rect in leaves:
        tris_list.extend(rect_to_triangles(rect, diag_mode, rng))

    tris = np.stack(tris_list, axis=0).astype(np.float32)  # (N,3,2)
    N = tris.shape[0]

    # Node features:
    # - vertices normalized (x/w, y/h) => 6
    # - centroid normalized => 2
    # - area normalized => 1
    # - orientation (sin, cos) => 2
    # - edge strength at centroid => 1
    Fnode = 6 + 2 + 1 + 2 + 1
    node_feat = np.zeros((N, Fnode), dtype=np.float32)
    colors_lin = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        tri = tris[i]
        tri_norm = tri.copy()
        tri_norm[:, 0] /= max(1.0, float(w))
        tri_norm[:, 1] /= max(1.0, float(h))

        ctr = triangle_centroid(tri)
        ctr_n = np.array([ctr[0] / max(1.0, w), ctr[1] / max(1.0, h)], dtype=np.float32)
        area = triangle_area(tri) / max(1.0, float(w * h))
        ang = triangle_orientation(tri)
        ang_sin = math.sin(ang)
        ang_cos = math.cos(ang)

        cx = int(np.clip(round(ctr[0]), 0, w - 1))
        cy = int(np.clip(round(ctr[1]), 0, h - 1))
        ectr = float(edge_map[cy, cx])

        feat = np.concatenate([
            tri_norm.reshape(-1),
            ctr_n,
            np.array([area], dtype=np.float32),
            np.array([ang_sin, ang_cos], dtype=np.float32),
            np.array([ectr], dtype=np.float32),
        ], axis=0)

        node_feat[i] = feat

        c_srgb = sample_triangle_color(img_rgb_u8, tri)  # sRGB float
        colors_lin[i] = srgb_to_lin(c_srgb)

    # Build shared-edge adjacency:
    # Each triangle has 3 local edges: (0-1),(1-2),(2-0)
    local_edges = [(0, 1), (1, 2), (2, 0)]
    edge_dict: Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}  # key -> (tri_idx, local_edge_idx)
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
                # shared edge between ti and tj
                i, j = (tj, ti)
                e_i = le2
                e_j = le

                # Compute edge features:
                # - shared edge length (normalized)
                # - distance between centroids (normalized)
                # - relative angle between triangle orientations (sin, cos)
                # - shared edge angle (sin, cos)
                tri_i = tris[i]
                tri_j = tris[j]
                ci = triangle_centroid(tri_i)
                cj = triangle_centroid(tri_j)
                dist = float(np.linalg.norm(ci - cj))
                dist_n = dist / max(1.0, float(max(w, h)))

                # shared edge endpoints (use current tri ti endpoints)
                p1 = tri[a].astype(np.float32)
                p2 = tri[b].astype(np.float32)
                elen = float(np.linalg.norm(p2 - p1))
                elen_n = elen / max(1.0, float(max(w, h)))

                eang = math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0]))
                eang_s = math.sin(eang)
                eang_c = math.cos(eang)

                ai = triangle_orientation(tri_i)
                aj = triangle_orientation(tri_j)
                da = aj - ai
                # wrap to [-pi,pi]
                da = (da + math.pi) % (2 * math.pi) - math.pi
                da_s = math.sin(da)
                da_c = math.cos(da)

                ef = np.array([elen_n, dist_n, da_s, da_c, eang_s, eang_c], dtype=np.float32)

                pairs.append((i, j))
                tri_edge_map.append((e_i, e_j))
                edge_feat_list.append(ef)

                # prevent double pairing for same key
                del edge_dict[key]

    if len(pairs) == 0:
        # Fallback: no shared edges found (shouldn't happen with grid-based splitting)
        edges = np.zeros((0, 2), dtype=np.int64)
        edge_feat = np.zeros((0, 6), dtype=np.float32)
        tri_edge_map_np = np.zeros((0, 2), dtype=np.int64)
    else:
        edges = np.array(pairs, dtype=np.int64)
        edge_feat = np.stack(edge_feat_list, axis=0).astype(np.float32)
        tri_edge_map_np = np.array(tri_edge_map, dtype=np.int64)

    return TriangleGraph(
        width=w,
        height=h,
        tris=tris,
        node_feat=node_feat,
        colors_lin=colors_lin,
        edges=edges,
        edge_feat=edge_feat,
        tri_edge_map=tri_edge_map_np,
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

    # linear -> srgb
    c = lin_to_srgb(np.clip(colors_lin, 0.0, 1.0))
    c_u8 = float_to_u8(c)

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
    """
    Exports a dataset zip:
      - graphs/<name>.json with triangles, node_feat, colors_lin, edges, edge_feat, tri_edge_map
    """
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
# -------------------- Torch: Graph Diffusion -----------------
# ============================================================

def sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) in [0,1]
    returns (B,dim)
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=device))
    # (B,half)
    ang = t[:, None] * freqs[None, :] * 2.0 * math.pi
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=1)
    return emb


class EdgeMessage(nn.Module):
    def __init__(self, hdim: int, edim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hdim * 2 + edim, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.SiLU(),
        )

    def forward(self, hi: torch.Tensor, hj: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([hi, hj, e], dim=-1))


class SimpleGNN(nn.Module):
    """
    Minimal message-passing GNN without torch_geometric:
    - node states h
    - for each undirected edge (i,j), send messages both ways
    """
    def __init__(self, node_in: int, edge_in: int, time_dim: int = 32, hdim: int = 128, layers: int = 4):
        super().__init__()
        self.time_dim = time_dim
        self.hdim = hdim
        self.layers = layers

        self.node_proj = nn.Linear(node_in, hdim)
        self.color_proj = nn.Linear(3, hdim)
        self.time_proj = nn.Linear(time_dim, hdim)

        self.msg = nn.ModuleList([EdgeMessage(hdim, edge_in) for _ in range(layers)])
        self.upd = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hdim * 2, hdim),
                nn.SiLU(),
                nn.Linear(hdim, hdim),
                nn.SiLU(),
            )
            for _ in range(layers)
        ])

        self.out = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.SiLU(),
            nn.Linear(hdim, 3),
        )

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor, x_noisy: torch.Tensor, t01: torch.Tensor) -> torch.Tensor:
        """
        node_feat: (N,F)
        edge_index: (E,2) long
        edge_feat: (E,Fe)
        x_noisy: (N,3) linear rgb
        t01: (N,) in [0,1]
        returns predicted noise eps_hat: (N,3)
        """
        h = self.node_proj(node_feat) + self.color_proj(x_noisy) + self.time_proj(sinusoidal_time_embed(t01, self.time_dim))

        if edge_index.numel() == 0:
            return self.out(h)

        i = edge_index[:, 0]
        j = edge_index[:, 1]

        for k in range(self.layers):
            # messages i<-j and j<-i
            mij = self.msg[k](h[i], h[j], edge_feat)  # (E,hdim)
            mji = self.msg[k](h[j], h[i], edge_feat)

            agg = torch.zeros_like(h)
            agg.index_add_(0, i, mij)
            agg.index_add_(0, j, mji)

            h = h + self.upd[k](torch.cat([h, agg], dim=-1))

        return self.out(h)


@dataclass
class DiffusionSchedule:
    T: int
    beta_start: float
    beta_end: float

    def make(self, device: torch.device):
        betas = torch.linspace(self.beta_start, self.beta_end, self.T, device=device)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        return betas, alphas, abar


def q_sample(x0: torch.Tensor, t_idx: torch.Tensor, abar: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    x_t = sqrt(a_bar) * x0 + sqrt(1-a_bar) * noise
    """
    a = abar[t_idx].view(-1, 1)
    return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise


@torch.no_grad()
def p_sample_loop(
    model: SimpleGNN,
    node_feat: torch.Tensor,
    edge_index: torch.Tensor,
    edge_feat: torch.Tensor,
    sched: DiffusionSchedule,
    device: torch.device,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """
    Samples colors from pure noise using DDPM-like steps.
    Returns x0 estimate (linear RGB in [0,1] but may exceed slightly; clamp later).
    """
    model.eval()
    betas, alphas, abar = sched.make(device)

    N = node_feat.shape[0]
    x = torch.randn((N, 3), device=device)

    for t in reversed(range(sched.T)):
        t_idx = torch.full((N,), t, device=device, dtype=torch.long)
        t01 = (t_idx.float() / max(1, sched.T - 1)).clamp(0, 1)

        eps = model(node_feat, edge_index, edge_feat, x, t01)  # predicted noise

        # Optional simple "guidance": just scale eps (acts like stronger denoise)
        eps = eps * float(guidance_scale)

        beta = betas[t]
        alpha = alphas[t]
        a_bar = abar[t]

        # DDPM mean:
        # mu = 1/sqrt(alpha) * (x - beta/sqrt(1-a_bar) * eps)
        mu = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - a_bar + 1e-8)) * eps)

        if t > 0:
            z = torch.randn_like(x)
            sigma = torch.sqrt(beta)
            x = mu + sigma * z
        else:
            x = mu

    return x


# ============================================================
# -------------------- Torch: Puzzle Edge Embedding -----------
# ============================================================

def triangle_local_edge_features(tri: np.ndarray, local_edge_idx: int, w: int, h: int) -> np.ndarray:
    """
    Features for "triangle+one edge" used by puzzle embedding:
      - edge length normalized
      - edge angle sin/cos
      - edge midpoint relative to centroid (dx,dy normalized)
      - triangle area normalized
      - triangle orientation sin/cos
    """
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


class EdgeTower(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class PuzzleMatcher(nn.Module):
    """
    Two-tower embedding; shared weights is typical and works well for matching.
    """
    def __init__(self, in_dim: int, emb_dim: int = 64):
        super().__init__()
        self.tower = EdgeTower(in_dim, emb_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower(x)


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """
    anchor: (B,D)
    positive: (B,D)
    negatives: (B,K,D)
    """
    # cosine sims
    pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True) / temp  # (B,1)
    neg_sim = torch.sum(anchor[:, None, :] * negatives, dim=-1) / temp  # (B,K)
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B,1+K)
    labels = torch.zeros((anchor.shape[0],), device=anchor.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


# ============================================================
# -------------------- Dataset Packing / Batching -------------
# ============================================================

@dataclass
class PackedGraph:
    node_feat: torch.Tensor        # (N,F)
    colors_lin: torch.Tensor       # (N,3)
    edge_index: torch.Tensor      # (E,2)
    edge_feat: torch.Tensor       # (E,Fe)
    width: int
    height: int
    tris: np.ndarray              # keep numpy for rendering


def pack_graph(g: TriangleGraph, device: torch.device) -> PackedGraph:
    return PackedGraph(
        node_feat=torch.from_numpy(g.node_feat).to(device),
        colors_lin=torch.from_numpy(g.colors_lin).to(device),
        edge_index=torch.from_numpy(g.edges).to(device).long(),
        edge_feat=torch.from_numpy(g.edge_feat).to(device),
        width=g.width,
        height=g.height,
        tris=g.tris,
    )


def batch_graphs(graphs: List[PackedGraph]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Concatenate multiple graphs into one big graph batch (disconnected components).
    Returns: node_feat, colors, edge_index, edge_feat, node_offsets
    """
    node_offsets = []
    n_total = 0
    for g in graphs:
        node_offsets.append(n_total)
        n_total += g.node_feat.shape[0]

    node_feat = torch.cat([g.node_feat for g in graphs], dim=0)
    colors = torch.cat([g.colors_lin for g in graphs], dim=0)

    edges_all = []
    efeat_all = []
    for off, g in zip(node_offsets, graphs):
        if g.edge_index.numel() == 0:
            continue
        edges_all.append(g.edge_index + off)
        efeat_all.append(g.edge_feat)

    if len(edges_all) == 0:
        edge_index = torch.zeros((0, 2), device=node_feat.device, dtype=torch.long)
        edge_feat = torch.zeros((0, graphs[0].edge_feat.shape[1] if graphs else 6), device=node_feat.device)
    else:
        edge_index = torch.cat(edges_all, dim=0)
        edge_feat = torch.cat(efeat_all, dim=0)

    return node_feat, colors, edge_index, edge_feat, node_offsets


# ============================================================
# --------------------------- Streamlit UI --------------------
# ============================================================

st.set_page_config(page_title="TriangleGraph Studio", page_icon="🔺", layout="wide")
st.title("🔺 TriangleGraph Studio — Diffusion + Puzzle AI")
st.caption(
    "Build triangle graphs from images, export datasets, train a graph diffusion model for color generation, "
    "and train a puzzle edge-matching model that learns which triangle edges belong together."
)

# Device selection (Streamlit Cloud often uses CPU)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

with st.sidebar:
    st.subheader("Compute")
    st.write(f"Device: **{device}**")
    if use_cuda:
        st.write("CUDA detected ✅")
    else:
        st.write("CPU mode (still works; keep graphs smaller)")

tabs = st.tabs(["1) Build Graphs", "2) Train Models", "3) Generate (Diffusion)", "4) Puzzle Mode (Matching)"])


# ============================================================
# Tab 1: Build graphs + export
# ============================================================
with tabs[0]:
    st.subheader("1) Build triangle graphs from images")
    st.write(
        "Upload one or more images. Each becomes a **triangle graph**:\n"
        "- nodes = triangles (variable sizes via adaptive quadtree)\n"
        "- edges = shared borders between triangles\n"
        "- node targets = triangle colors (linear RGB) sampled from the image\n"
    )

    uploads = st.file_uploader(
        "Upload images (PNG/JPG/WebP) — you can upload multiple for training",
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
            previews = []

            with st.spinner("Building triangle graphs…"):
                for f in uploads:
                    img = Image.open(io.BytesIO(f.getvalue()))
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    img = resize_keep_aspect(img, int(max_side))
                    rgb_u8 = pil_to_np_rgb(img)

                    g = build_triangle_graph(rgb_u8, qp, diag_mode=diag_mode, seed=int(seed_graph))
                    graphs.append(g)
                    base = f.name.rsplit(".", 1)[0]
                    names.append(base)

                    # preview mosaic
                    preview_img = render_triangle_mosaic(
                        g,
                        g.colors_lin,
                        gap_px=float(preview_gap),
                        background=preview_bg,
                        outline=preview_outline,
                        outline_px=int(outline_px),
                        outline_alpha=float(outline_alpha),
                    )
                    previews.append((f.name, preview_img, g.tris.shape[0], g.edges.shape[0], g.width, g.height))

            # persist in session
            st.session_state["graphs"] = graphs
            st.session_state["graph_names"] = names

            st.success(f"Built {len(graphs)} graph(s).")

            for nm, pimg, ntri, ned, w, h in previews:
                st.markdown(f"**{nm}** — triangles: **{ntri}**, edges: **{ned}**, size: **{w}×{h}**")
                st.image(pimg, use_container_width=True)

            z = export_graph_zip(graphs, names)
            st.download_button(
                "Download dataset ZIP (graphs/*.json + index.json)",
                data=z,
                file_name="triangle_graph_dataset.zip",
                mime="application/zip",
                use_container_width=True,
            )


# ============================================================
# Tab 2: Train models
# ============================================================
with tabs[1]:
    st.subheader("2) Train models")
    st.write(
        "This trains two models **on your uploaded graphs**:\n"
        "1) **Graph Diffusion Denoiser** — learns to denoise triangle colors using adjacency.\n"
        "2) **Puzzle Edge Matcher** — learns edge compatibility (contrastive) to suggest neighbors.\n"
        "\nTip: Start with 1–3 images, keep triangle counts modest (< ~1500 triangles) for CPU."
    )

    if "graphs" not in st.session_state:
        st.info("Go to **1) Build Graphs** first.")
    else:
        graphs_np: List[TriangleGraph] = st.session_state["graphs"]
        names = st.session_state["graph_names"]

        # Convert to torch packed graphs
        packed = [pack_graph(g, device=device) for g in graphs_np]

        colA, colB = st.columns([1, 1], gap="large")

        with colA:
            st.markdown("### Diffusion model training")
            T = st.slider("Diffusion steps (T)", 20, 200, 80, 5)
            beta_start = st.slider("beta_start", 1e-5, 5e-3, 5e-4, format="%.6f")
            beta_end = st.slider("beta_end", 1e-3, 5e-2, 2e-2, format="%.6f")
            hdim = st.selectbox("GNN hidden dim", [64, 96, 128, 160, 192], index=2)
            layers = st.slider("GNN layers", 2, 8, 4, 1)

            epochs = st.slider("Epochs", 1, 50, 10, 1)
            lr = st.number_input("Learning rate", min_value=1e-5, max_value=5e-2, value=2e-3, format="%.6f")
            steps_per_epoch = st.slider("Steps per epoch", 10, 500, 120, 10)
            grad_clip = st.slider("Grad clip", 0.1, 5.0, 1.0, 0.1)

        with colB:
            st.markdown("### Puzzle matcher training")
            emb_dim = st.selectbox("Embedding dim", [32, 48, 64, 96, 128], index=2)
            pm_epochs = st.slider("Matcher epochs", 1, 50, 8, 1)
            pm_lr = st.number_input("Matcher LR", min_value=1e-5, max_value=5e-2, value=2e-3, format="%.6f", key="pm_lr")
            batch_size = st.slider("Edge-batch size", 32, 1024, 256, 32)
            negatives_k = st.slider("Negatives per anchor", 8, 128, 32, 8)
            temp = st.slider("InfoNCE temperature", 0.02, 0.2, 0.07, 0.01)

        train_btn = st.button("Train both models", type="primary", use_container_width=True)

        if train_btn:
            # -----------------
            # Diffusion training
            # -----------------
            node_feat, colors, edge_index, edge_feat, node_offsets = batch_graphs(packed)

            sched = DiffusionSchedule(T=int(T), beta_start=float(beta_start), beta_end=float(beta_end))
            betas, alphas, abar = sched.make(device)

            model = SimpleGNN(
                node_in=node_feat.shape[1],
                edge_in=edge_feat.shape[1] if edge_feat.numel() else 6,
                time_dim=32,
                hdim=int(hdim),
                layers=int(layers),
            ).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

            def diffusion_step():
                # sample random time indices per node
                t_idx = torch.randint(low=0, high=sched.T, size=(node_feat.shape[0],), device=device)
                t01 = (t_idx.float() / max(1, sched.T - 1)).clamp(0, 1)

                noise = torch.randn_like(colors)
                x_t = q_sample(colors, t_idx, abar, noise)

                eps_hat = model(node_feat, edge_index, edge_feat, x_t, t01)
                loss = F.mse_loss(eps_hat, noise)
                return loss

            st.write("#### Training diffusion denoiser…")
            prog = st.progress(0)
            losses = []
            t0 = time.time()

            model.train()
            for ep in range(int(epochs)):
                ep_losses = []
                for s in range(int(steps_per_epoch)):
                    opt.zero_grad(set_to_none=True)
                    loss = diffusion_step()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    opt.step()
                    ep_losses.append(float(loss.item()))
                losses.append(float(np.mean(ep_losses)))
                prog.progress(int((ep + 1) / int(epochs) * 100))
                st.write(f"Epoch {ep+1}/{epochs} — loss: {losses[-1]:.5f}")

            st.success(f"Diffusion training done in {time.time() - t0:.1f}s.")
            st.session_state["diffusion_model_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            st.session_state["diffusion_config"] = {
                "T": int(T),
                "beta_start": float(beta_start),
                "beta_end": float(beta_end),
                "node_in": int(node_feat.shape[1]),
                "edge_in": int(edge_feat.shape[1] if edge_feat.numel() else 6),
                "hdim": int(hdim),
                "layers": int(layers),
            }

            # -----------------
            # Puzzle matcher training
            # -----------------
            # Build training pairs from shared edges in each graph:
            # each graph edge gives a positive pair of (tri i, local_edge_i) and (tri j, local_edge_j)
            st.write("#### Training puzzle edge matcher…")
            pm_in_dim = 8  # triangle+edge features defined above
            matcher = PuzzleMatcher(in_dim=pm_in_dim, emb_dim=int(emb_dim)).to(device)
            pm_opt = torch.optim.AdamW(matcher.parameters(), lr=float(pm_lr))

            # Gather all samples across graphs
            samples_A = []
            samples_B = []

            for g in graphs_np:
                if g.edges.shape[0] == 0:
                    continue
                for e_idx in range(g.edges.shape[0]):
                    i, j = int(g.edges[e_idx, 0]), int(g.edges[e_idx, 1])
                    ei, ej = int(g.tri_edge_map[e_idx, 0]), int(g.tri_edge_map[e_idx, 1])

                    fa = triangle_local_edge_features(g.tris[i], ei, g.width, g.height)
                    fb = triangle_local_edge_features(g.tris[j], ej, g.width, g.height)
                    samples_A.append(fa)
                    samples_B.append(fb)

                    # also add reverse direction as another positive
                    samples_A.append(fb)
                    samples_B.append(fa)

            if len(samples_A) < 64:
                st.warning("Not enough shared edges to train matcher well. Use settings that create more triangles.")
            else:
                A = torch.from_numpy(np.stack(samples_A, axis=0)).to(device)
                B = torch.from_numpy(np.stack(samples_B, axis=0)).to(device)

                n = A.shape[0]
                prog2 = st.progress(0)
                matcher.train()
                t1 = time.time()

                for ep in range(int(pm_epochs)):
                    # shuffle indices
                    idx = torch.randperm(n, device=device)
                    ep_losses = []

                    for s in range(0, n, int(batch_size)):
                        batch_idx = idx[s:s + int(batch_size)]
                        if batch_idx.numel() < 8:
                            continue

                        anc = A[batch_idx]
                        pos = B[batch_idx]

                        # negatives: choose random other Bs
                        # (B,K,D) after embedding
                        neg_idx = torch.randint(0, n, (batch_idx.shape[0], int(negatives_k)), device=device)
                        neg = B[neg_idx]  # (B,K,in)

                        pm_opt.zero_grad(set_to_none=True)
                        z_anc = matcher.embed(anc)
                        z_pos = matcher.embed(pos)
                        z_neg = matcher.embed(neg.view(-1, pm_in_dim)).view(batch_idx.shape[0], int(negatives_k), -1)
                        loss = info_nce_loss(z_anc, z_pos, z_neg, temp=float(temp))
                        loss.backward()
                        pm_opt.step()
                        ep_losses.append(float(loss.item()))

                    prog2.progress(int((ep + 1) / int(pm_epochs) * 100))
                    st.write(f"Epoch {ep+1}/{pm_epochs} — loss: {float(np.mean(ep_losses)):.5f}")

                st.success(f"Matcher training done in {time.time() - t1:.1f}s.")
                st.session_state["matcher_state"] = {k: v.detach().cpu() for k, v in matcher.state_dict().items()}
                st.session_state["matcher_config"] = {"in_dim": pm_in_dim, "emb_dim": int(emb_dim)}


# ============================================================
# Tab 3: Generate with diffusion
# ============================================================
with tabs[2]:
    st.subheader("3) Generate new triangle colorings (Graph Diffusion)")
    st.write(
        "Pick an existing triangle graph (geometry + adjacency). The diffusion model generates a **new image** "
        "by sampling triangle colors conditioned on neighborhood structure."
    )

    if "graphs" not in st.session_state:
        st.info("Go to **1) Build Graphs** first.")
    elif "diffusion_model_state" not in st.session_state:
        st.info("Train the diffusion model in **2) Train Models** first.")
    else:
        graphs_np: List[TriangleGraph] = st.session_state["graphs"]
        names = st.session_state["graph_names"]

        pick = st.selectbox("Choose graph geometry", names, index=0)
        gi = names.index(pick)
        g = graphs_np[gi]

        gen_gap = st.slider("Gap (px)", 0.0, 30.0, 8.0, 0.5)
        gen_bg = st.radio("Background", ["White", "Black"], horizontal=True)
        gen_outline = st.checkbox("Outlines", value=False)
        gen_outline_px = st.slider("Outline width", 1, 5, 1, 1) if gen_outline else 1
        gen_outline_alpha = st.slider("Outline opacity", 0.05, 1.0, 0.25, 0.01) if gen_outline else 0.25

        guidance = st.slider("Denoise strength (guidance)", 0.5, 2.5, 1.1, 0.05)

        gen_btn = st.button("Generate (sample)", type="primary", use_container_width=True)

        # show original
        st.markdown("**Original (triangle mosaic colors from image):**")
        st.image(
            render_triangle_mosaic(g, g.colors_lin, gap_px=float(gen_gap), background=gen_bg,
                                   outline=gen_outline, outline_px=int(gen_outline_px), outline_alpha=float(gen_outline_alpha)),
            use_container_width=True
        )

        if gen_btn:
            cfg = st.session_state["diffusion_config"]
            sched = DiffusionSchedule(T=int(cfg["T"]), beta_start=float(cfg["beta_start"]), beta_end=float(cfg["beta_end"]))

            model = SimpleGNN(
                node_in=int(cfg["node_in"]),
                edge_in=int(cfg["edge_in"]),
                time_dim=32,
                hdim=int(cfg["hdim"]),
                layers=int(cfg["layers"]),
            ).to(device)
            model.load_state_dict({k: v.to(device) for k, v in st.session_state["diffusion_model_state"].items()})
            model.eval()

            pg = pack_graph(g, device=device)

            with st.spinner("Sampling diffusion…"):
                x = p_sample_loop(
                    model=model,
                    node_feat=pg.node_feat,
                    edge_index=pg.edge_index,
                    edge_feat=pg.edge_feat,
                    sched=sched,
                    device=device,
                    guidance_scale=float(guidance),
                )
                # clamp to [0,1] for display
                x = torch.clamp(x, 0.0, 1.0).detach().cpu().numpy().astype(np.float32)

            out_img = render_triangle_mosaic(
                g, x, gap_px=float(gen_gap), background=gen_bg,
                outline=gen_outline, outline_px=int(gen_outline_px), outline_alpha=float(gen_outline_alpha)
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


# ============================================================
# Tab 4: Puzzle Mode with matcher suggestions
# ============================================================
with tabs[3]:
    st.subheader("4) Puzzle Mode — edge matching suggestions (Snap-Together AI)")
    st.write(
        "This mode treats the triangles like a **jigsaw**. The matcher model suggests which triangle edge "
        "is most compatible with your selected edge.\n\n"
        "Workflow:\n"
        "1) Choose a graph.\n"
        "2) Choose a triangle ID and edge (0/1/2).\n"
        "3) Get top neighbor suggestions.\n"
    )

    if "graphs" not in st.session_state:
        st.info("Go to **1) Build Graphs** first.")
    elif "matcher_state" not in st.session_state:
        st.info("Train the matcher in **2) Train Models** first.")
    else:
        graphs_np: List[TriangleGraph] = st.session_state["graphs"]
        names = st.session_state["graph_names"]

        pick = st.selectbox("Choose puzzle image graph", names, index=0, key="puzzle_pick")
        gi = names.index(pick)
        g = graphs_np[gi]

        # Build matcher model
        mcfg = st.session_state["matcher_config"]
        matcher = PuzzleMatcher(in_dim=int(mcfg["in_dim"]), emb_dim=int(mcfg["emb_dim"])).to(device)
        matcher.load_state_dict({k: v.to(device) for k, v in st.session_state["matcher_state"].items()})
        matcher.eval()

        puzzle_gap = st.slider("Puzzle gap (px)", 0.0, 35.0, 14.0, 0.5)
        puzzle_bg = st.radio("Background", ["White", "Black"], horizontal=True, key="puzzle_bg")
        show_colors = st.checkbox("Show colors (turn OFF for harder puzzle)", value=True)

        # Show current puzzle board (assembled but separated)
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

        suggest_btn = st.button("Suggest matches", type="primary", use_container_width=True)

        # Precompute all edge-embeddings for this graph for fast matching:
        @st.cache_data(show_spinner=False)
        def _precompute_edge_bank(tris: np.ndarray, w: int, h: int) -> np.ndarray:
            feats = []
            ids = []
            for tid in range(tris.shape[0]):
                for e in range(3):
                    feats.append(triangle_local_edge_features(tris[tid], e, w, h))
                    ids.append((tid, e))
            return np.stack(feats, axis=0).astype(np.float32), ids

        bank_feats_np, bank_ids = _precompute_edge_bank(g.tris, g.width, g.height)

        if suggest_btn:
            with st.spinner("Computing edge match suggestions…"):
                q_feat = triangle_local_edge_features(g.tris[int(tri_id)], int(edge_id), g.width, g.height)
                q = torch.from_numpy(q_feat[None, :]).to(device)
                zq = matcher.embed(q)  # (1,D)

                bank = torch.from_numpy(bank_feats_np).to(device)
                zb = matcher.embed(bank)  # (M,D)

                # similarity
                sim = (zq @ zb.t()).squeeze(0)  # (M,)

                # exclude self edge
                mask = torch.ones_like(sim, dtype=torch.bool)
                for k, (tid, eid) in enumerate(bank_ids):
                    if tid == int(tri_id) and eid == int(edge_id):
                        mask[k] = False
                        break
                sim2 = sim.clone()
                sim2[~mask] = -1e9

                vals, idx = torch.topk(sim2, k=int(topk))
                idx = idx.detach().cpu().numpy()
                vals = vals.detach().cpu().numpy()

            st.markdown("### Top matches")
            for rank, (kidx, s) in enumerate(zip(idx, vals), start=1):
                tid, eid = bank_ids[int(kidx)]
                st.write(f"{rank}. Triangle **{tid}**, edge **{eid}** — similarity: **{float(s):.4f}**")

            st.info(
                "In a full interactive jigsaw, you’d snap the suggested edge onto your selected edge. "
                "This app gives you the AI’s best guesses; you can expand this into drag-and-snap later."
            )


# ============================================================
# Footer: quick explanation for you
# ============================================================
with st.expander("What you can extend next (recommended)", expanded=False):
    st.write(
        "- Add a **true assembly solver**: greedy matching + cycle checks to reconstruct adjacency from shuffled pieces.\n"
        "- Add **conditional diffusion**: let user paint a few triangle colors as constraints; diffusion fills the rest.\n"
        "- Add **style embeddings**: train on multiple images grouped by style; condition diffusion on style id.\n"
        "- Add **SVG export** of triangles for real physical puzzles or stained glass templates.\n"
    )
