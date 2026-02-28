import io, json, math, hashlib, time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

# ============================================================
# TriangleGraph Text-to-Image (No Torch) — v2 (Best effort)
# ============================================================
# Goals:
# - Better prompt conditioning (palette + structure + lighting)
# - Better "stained glass" look (soft quantization + micro texture)
# - Edge-aware anisotropic smoothing on triangle-graph
# - Training (A/B): preference model + optional auto-tuning defaults
# - Text tab generates ONE image per click (low crash risk)
# ============================================================

# ------------------ Utility ------------------

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def float_to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

def u8_to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 255.0

def srgb_to_lin(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)

def lin_to_srgb(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.0031308, 12.92 * c, (1 + a) * (c ** (1 / 2.4)) - a)

def hex_to_rgb01(h: str) -> np.ndarray:
    h = h.strip().lstrip("#")
    if len(h) != 6:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return np.array([r, g, b], dtype=np.float32) / 255.0

def stable_hash_int(s: str, mod: int = 2**31-1) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % mod

def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / max(w, h)
    return img.resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS)

# ------------------ Geometry / Graph ------------------

def tri_centroid(t: np.ndarray) -> np.ndarray:
    return np.mean(t.astype(np.float32), axis=0)

def tri_min_edge(t: np.ndarray) -> float:
    p = t.astype(np.float32)
    a, b, c = p
    ab = float(np.linalg.norm(a-b))
    bc = float(np.linalg.norm(b-c))
    ca = float(np.linalg.norm(c-a))
    return max(1e-6, min(ab, bc, ca))

def shrink_triangle(pts: np.ndarray, gap_px: float) -> np.ndarray:
    pts = pts.astype(np.float32)
    ctr = tri_centroid(pts)
    m = tri_min_edge(pts)
    scale = 1.0 - float(gap_px) / m
    scale = max(0.05, min(1.0, scale))
    return ctr[None, :] + (pts - ctr[None, :]) * scale

def _edge_key(p1: np.ndarray, p2: np.ndarray) -> Tuple[int,int,int,int]:
    a = (int(round(float(p1[0]))), int(round(float(p1[1]))))
    b = (int(round(float(p2[0]))), int(round(float(p2[1]))))
    if a <= b:
        return (a[0], a[1], b[0], b[1])
    return (b[0], b[1], a[0], a[1])

@dataclass
class TriangleGraph:
    width: int
    height: int
    tris: np.ndarray          # (N,3,2) float32
    edges: np.ndarray         # (E,2) int64
    edge_strength: np.ndarray # (N,) float32 0..1

def build_edges_from_tris(tris: np.ndarray) -> np.ndarray:
    local_edges = [(0,1),(1,2),(2,0)]
    edge_dict = {}
    pairs = []
    for ti in range(tris.shape[0]):
        t = tris[ti]
        for (a,b) in local_edges:
            key = _edge_key(t[a], t[b])
            if key not in edge_dict:
                edge_dict[key] = ti
            else:
                tj = edge_dict[key]
                if tj != ti:
                    pairs.append((tj, ti))
                del edge_dict[key]
    if not pairs:
        return np.zeros((0,2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)

def build_neighbor_lists(edges: np.ndarray, N: int) -> List[List[int]]:
    nbr = [[] for _ in range(N)]
    for i,j in edges.astype(np.int64):
        nbr[int(i)].append(int(j)); nbr[int(j)].append(int(i))
    return nbr

# ------------------ Procedural triangulation ------------------

@dataclass
class ProcParams:
    min_cell: int
    max_depth: int
    split_prob: float
    diag_mode: str

def rect_to_tris(rect: Tuple[int,int,int,int], diag_mode: str, rng: np.random.Generator) -> List[np.ndarray]:
    x0,y0,x1,y1 = rect
    p00 = np.array([x0,y0], np.float32)
    p10 = np.array([x1,y0], np.float32)
    p01 = np.array([x0,y1], np.float32)
    p11 = np.array([x1,y1], np.float32)

    mode = diag_mode
    if diag_mode == "Random":
        mode = "TL-BR" if rng.random() < 0.5 else "TR-BL"
    elif diag_mode == "Alternate":
        mode = "TL-BR" if ((x0+y0) % 2 == 0) else "TR-BL"

    if mode == "TL-BR":
        t1 = np.stack([p00,p10,p11],0)
        t2 = np.stack([p00,p11,p01],0)
    else:
        t1 = np.stack([p00,p10,p01],0)
        t2 = np.stack([p10,p11,p01],0)
    return [t1,t2]

@st.cache_data(show_spinner=False, max_entries=64)
def build_procedural_graph(width: int, height: int, params: ProcParams, seed: int) -> TriangleGraph:
    rng = np.random.default_rng(int(seed))
    leaves: List[Tuple[int,int,int,int]] = []

    def rec(x0,y0,x1,y1,depth):
        rw, rh = x1-x0, y1-y0
        if rw <= params.min_cell or rh <= params.min_cell or depth >= params.max_depth:
            leaves.append((x0,y0,x1,y1)); return
        cx = (x0+x1)/2; cy = (y0+y1)/2
        spatial = 0.5 + 0.5*math.sin(0.015*cx + 0.02*cy + 2.0*rng.random())
        p = params.split_prob * (0.65 + 0.35*spatial)
        if rng.random() < p:
            mx = (x0+x1)//2; my = (y0+y1)//2
            if mx == x0 or mx == x1 or my == y0 or my == y1:
                leaves.append((x0,y0,x1,y1)); return
            rec(x0,y0,mx,my,depth+1)
            rec(mx,y0,x1,my,depth+1)
            rec(x0,my,mx,y1,depth+1)
            rec(mx,my,x1,y1,depth+1)
        else:
            leaves.append((x0,y0,x1,y1))

    rec(0,0,width,height,0)

    tris = []
    for r in leaves:
        tris.extend(rect_to_tris(r, params.diag_mode, rng))
    tris = np.stack(tris,0).astype(np.float32)

    edges = build_edges_from_tris(tris)
    ctrs = np.mean(tris, axis=1)  # (N,2)
    x = ctrs[:,0] / max(1.0,width)
    y = ctrs[:,1] / max(1.0,height)
    band1 = np.abs(np.sin(2*math.pi*(1.5*x + 0.7*y + 0.13)))
    band2 = np.abs(np.sin(2*math.pi*(0.6*x - 1.2*y + 0.41)))
    band3 = np.abs(np.sin(2*math.pi*(1.1*x + 1.1*y + 0.77)))
    edge_strength = (0.42*band1 + 0.33*band2 + 0.25*band3).astype(np.float32)
    edge_strength = np.clip(edge_strength, 0.0, 1.0)

    return TriangleGraph(width=width, height=height, tris=tris, edges=edges, edge_strength=edge_strength)

# ------------------ Rendering ------------------

def render_mosaic(g: TriangleGraph, colors_lin: np.ndarray, gap_px: float, background: str, outline: bool, outline_px: int, outline_alpha: float, lead_line_strength: float = 0.0) -> Image.Image:
    w,h = g.width, g.height
    bg = (255,255,255,255) if background == "White" else (0,0,0,255)
    canvas = Image.new("RGBA", (w,h), bg)
    draw = ImageDraw.Draw(canvas, "RGBA")

    srgb = lin_to_srgb(np.clip(colors_lin,0.0,1.0))
    cu8 = float_to_u8(srgb)

    outline_rgba = None
    if outline:
        oa = int(round(255*clamp01(outline_alpha)))
        outline_rgba = (0,0,0,oa) if background == "White" else (255,255,255,oa)

    # "lead lines" darker near strong edges
    lead = np.clip(g.edge_strength.astype(np.float32), 0.0, 1.0)
    lead = (lead_line_strength * lead).astype(np.float32)

    for i in range(g.tris.shape[0]):
        t = g.tris[i]
        ts = shrink_triangle(t, float(gap_px))
        poly = [(float(p[0]), float(p[1])) for p in ts]

        # darken color slightly if lead strength high
        c = cu8[i].astype(np.float32)/255.0
        c = np.clip(c*(1.0 - 0.25*lead[i]), 0.0, 1.0)
        fill = (int(c[0]*255+0.5), int(c[1]*255+0.5), int(c[2]*255+0.5), 255)
        draw.polygon(poly, fill=fill)

        if outline_rgba is not None and outline_px>0:
            draw.line(poly+[poly[0]], fill=outline_rgba, width=int(outline_px), joint="curve")

    return canvas

# ------------------ Prompt parsing ------------------

def parse_light(prompt: str) -> Tuple[float,float]:
    p = prompt.lower()
    dx, dy = -0.7, -0.7
    # explicit directional phrases
    if "light from left" in p or "from left" in p: dx, dy = -1.0, 0.0
    if "light from right" in p or "from right" in p: dx, dy = 1.0, 0.0
    if "light from above" in p or "from above" in p or "from top" in p: dx, dy = 0.0, -1.0
    if "light from below" in p or "from below" in p or "from bottom" in p: dx, dy = 0.0, 1.0
    if "top-right" in p: dx, dy = 0.7, -0.7
    if "bottom-right" in p: dx, dy = 0.7, 0.7
    if "bottom-left" in p: dx, dy = -0.7, 0.7
    return float(dx), float(dy)

def detect_style(model: Dict, prompt: str) -> str:
    p = prompt.lower()
    if "stained glass" in p or "glass" in p: return "stained_glass"
    if "mosaic" in p: return "mosaic"
    if "soft" in p or "dreamy" in p: return "soft"
    # fallback to defaults
    return str(model.get("defaults", {}).get("style", "stained_glass"))

def pick_palette(model: Dict, prompt: str) -> List[np.ndarray]:
    p = prompt.lower()
    pal = model.get("palettes", {})
    kw = model.get("prompt_keywords", {})

    key = None
    for k, words in kw.items():
        if any(w in p for w in words) and k in pal:
            key = k
            break
    if key is None:
        keys = list(pal.keys())
        if not keys:
            return [srgb_to_lin(np.array([0.5,0.5,0.5], np.float32))]
        key = keys[stable_hash_int(prompt) % len(keys)]

    return [srgb_to_lin(hex_to_rgb01(hx)) for hx in pal[key]]

# ------------------ Noise / multi-scale fields ------------------

def rand_unit_vec2(rng: np.random.Generator) -> Tuple[float,float]:
    a = float(rng.uniform(0.0, 2*math.pi))
    return math.cos(a), math.sin(a)

def fbm_noise_2d(x: np.ndarray, y: np.ndarray, seed: int, octaves: int = 4) -> np.ndarray:
    # lightweight fBm using sine/cosine warps (fast & stable)
    rng = np.random.default_rng(int(seed))
    out = np.zeros_like(x, dtype=np.float32)
    amp = 1.0
    freq = 1.0
    for _ in range(int(octaves)):
        ux, uy = rand_unit_vec2(rng)
        phase = float(rng.uniform(0.0, 10.0))
        out += amp * np.sin(2*math.pi*(freq*(ux*x + uy*y) + phase)).astype(np.float32)
        amp *= 0.5
        freq *= 2.0
    # normalize to 0..1
    mn = float(np.min(out)); mx = float(np.max(out))
    return ((out - mn) / (mx - mn + 1e-8)).astype(np.float32)

def prompt_field(g: TriangleGraph, palette: List[np.ndarray], prompt: str, seed: int, multi_scale: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    ctr = np.mean(g.tris, axis=1)  # (N,2)
    x = (ctr[:,0] / max(1.0,g.width)).astype(np.float32)
    y = (ctr[:,1] / max(1.0,g.height)).astype(np.float32)

    dx, dy = parse_light(prompt)
    proj = dx*(x-0.5) + dy*(y-0.5)
    proj = (proj - float(np.min(proj))) / (float(np.ptp(proj)) + 1e-8)

    # secondary axis & fBm texture
    proj2 = (x + 0.6*y + 0.17*np.sin(2*math.pi*(x-y))).astype(np.float32)
    proj2 = (proj2 - float(np.min(proj2))) / (float(np.ptp(proj2)) + 1e-8)

    n = fbm_noise_2d(x, y, seed=seed+777, octaves=4)
    t = (0.55*proj + 0.25*proj2 + float(multi_scale)*0.20*n).astype(np.float32)
    t = np.clip(t, 0.0, 0.999999)

    K = len(palette)
    pos = t*(K-1)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.clip(i0+1, 0, K-1)
    a = (pos - i0.astype(np.float32)).astype(np.float32)

    pal0 = np.stack([palette[i] for i in i0], 0)
    pal1 = np.stack([palette[i] for i in i1], 0)
    base = (1-a)[:,None]*pal0 + a[:,None]*pal1

    tex = rng.normal(0.0, 0.02, size=base.shape).astype(np.float32)
    return np.clip(base + tex, 0.0, 1.0).astype(np.float32)

# ------------------ Soft quantization to palette ------------------

def soft_quantize(colors: np.ndarray, palette: np.ndarray, strength: float) -> np.ndarray:
    """
    Pull colors toward palette via softmax weights (still smooth).
    strength: 0..1
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if palette.shape[0] < 2 or strength <= 1e-6:
        return colors
    # distance in linear RGB
    # weights = softmax(-d2 / tau)
    tau = 0.08 + 0.22*(1.0-strength)  # smaller tau => sharper to palette
    d2 = np.sum((colors[:,None,:] - palette[None,:,:])**2, axis=2)  # (N,K)
    w = np.exp(-d2 / (tau + 1e-8))
    w = w / (np.sum(w, axis=1, keepdims=True) + 1e-8)
    q = w @ palette  # (N,3)
    return (1.0-strength)*colors + strength*q

# ------------------ Generation Model (edge-aware anisotropic smoothing) ------------------

def generate_colors(
    model: Dict,
    g: TriangleGraph,
    prompt: str,
    seed: int,
    steps: int,
    neighbor_w: float,
    prompt_w: float,
    edge_preserve: float,
    temperature: float,
    multi_scale: float,
    quantize_strength: float,
    micro_texture: float,
    contrast_boost: float,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    N = g.tris.shape[0]
    nbr = build_neighbor_lists(g.edges, N)

    palette_list = pick_palette(model, prompt)
    palette = np.stack(palette_list, axis=0).astype(np.float32)

    pf = prompt_field(g, palette_list, prompt, seed=seed+1234, multi_scale=multi_scale)

    # init with prompt-biased noise
    x = np.clip(pf + rng.normal(0.0, 0.22*temperature, size=(N,3)).astype(np.float32), 0.0, 1.0)

    es = np.clip(g.edge_strength.astype(np.float32), 0.0, 1.0)
    resist = float(edge_preserve) * es  # 0..edge_preserve

    # micro texture (glass grain): fixed per triangle, not per step
    grain = rng.normal(0.0, 1.0, size=(N,3)).astype(np.float32)
    grain = grain / (np.std(grain) + 1e-6)

    for t in range(int(steps)):
        x_new = x.copy()

        # neighbor mean + anisotropic mixing
        for i in range(N):
            ns = nbr[i]
            if ns:
                m = np.mean(x[ns], axis=0)
                # anisotropic: weigh by similarity to prevent bleeding across "edges"
                # compute similarity weights quickly (few neighbors)
                arr = x[ns]
                d2 = np.sum((arr - x[i][None,:])**2, axis=1)
                wsim = np.exp(-d2 / 0.06)
                wsim = wsim / (np.sum(wsim) + 1e-8)
                m = (wsim[:,None] * arr).sum(axis=0)
            else:
                m = x[i]

            w_smooth = float(neighbor_w) * (1.0 - resist[i])
            x_new[i] = (1.0 - w_smooth - float(prompt_w)) * x[i] + w_smooth * m + float(prompt_w) * pf[i]

        # edge-driven contrast: push apart on strong-edge connections
        if g.edges.shape[0] > 0 and contrast_boost > 1e-6:
            i = g.edges[:,0].astype(np.int64); j = g.edges[:,1].astype(np.int64)
            d = x_new[i] - x_new[j]
            push = (float(contrast_boost) * 0.35 * (es[i] + es[j]) / 2.0)[:,None]
            x_new[i] = np.clip(x_new[i] + push * d, 0.0, 1.0)
            x_new[j] = np.clip(x_new[j] - push * d, 0.0, 1.0)

        # mild annealed noise
        sigma = (0.018 * temperature) * (1.0 - (t / max(1, steps-1)) * 0.65)
        x_new = np.clip(x_new + rng.normal(0.0, sigma, size=x_new.shape).astype(np.float32), 0.0, 1.0)

        # soft quantize every few steps to pull toward palette (stained glass feel)
        if (t % 4) == 0 and quantize_strength > 1e-6:
            x_new = np.clip(soft_quantize(x_new, palette, strength=quantize_strength), 0.0, 1.0)

        x = x_new

    # add micro texture at end (subtle)
    if micro_texture > 1e-6:
        x = np.clip(x + float(micro_texture) * 0.06 * grain, 0.0, 1.0)

    return x.astype(np.float32)

# ------------------ Image -> Triangle Mosaic (optional) ------------------

def sobel_edge_strength(rgb_u8: np.ndarray) -> np.ndarray:
    x = u8_to_float(rgb_u8)
    lum = 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32)
    p = np.pad(lum, 1, mode="edge")
    gx = (
        kx[0,0]*p[:-2,:-2] + kx[0,1]*p[:-2,1:-1] + kx[0,2]*p[:-2,2:] +
        kx[1,0]*p[1:-1,:-2] + kx[1,1]*p[1:-1,1:-1] + kx[1,2]*p[1:-1,2:] +
        kx[2,0]*p[2:,:-2] + kx[2,1]*p[2:,1:-1] + kx[2,2]*p[2:,2:]
    )
    gy = (
        ky[0,0]*p[:-2,:-2] + ky[0,1]*p[:-2,1:-1] + ky[0,2]*p[:-2,2:] +
        ky[1,0]*p[1:-1,:-2] + ky[1,1]*p[1:-1,1:-1] + ky[1,2]*p[1:-1,2:] +
        ky[2,0]*p[2:,:-2] + ky[2,1]*p[2:,1:-1] + ky[2,2]*p[2:,2:]
    )
    mag = np.sqrt(gx*gx + gy*gy)
    hi = float(np.quantile(mag, 0.995)) + 1e-8
    return np.clip(mag/hi, 0.0, 1.0).astype(np.float32)

@dataclass
class QuadParams:
    min_cell: int
    max_depth: int
    var_thresh: float
    edge_thresh: float
    edge_weight: float
    diag_mode: str

def region_stats(rgb_u8: np.ndarray, edge: np.ndarray, x0:int,y0:int,x1:int,y1:int) -> Tuple[float,float]:
    roi = u8_to_float(rgb_u8[y0:y1, x0:x1])
    if roi.size == 0:
        return 0.0, 0.0
    var = float(np.mean(np.var(roi.reshape(-1,3), axis=0)))
    em = float(np.mean(edge[y0:y1, x0:x1]))
    return var, em

def build_quadtree_leaves(rgb_u8: np.ndarray, edge: np.ndarray, qp: QuadParams) -> List[Tuple[int,int,int,int]]:
    h,w,_ = rgb_u8.shape
    leaves = []
    def should_split(x0,y0,x1,y1,depth):
        rw, rh = x1-x0, y1-y0
        if rw <= qp.min_cell or rh <= qp.min_cell: return False
        if depth >= qp.max_depth: return False
        var, em = region_stats(rgb_u8, edge, x0,y0,x1,y1)
        ew = clamp01(qp.edge_weight)
        var_n = min(1.0, var/0.03)
        em_n = float(em)
        score = (1-ew)*var_n + ew*em_n
        return (var_n > qp.var_thresh) or (em_n > qp.edge_thresh) or (score > max(qp.var_thresh, qp.edge_thresh))
    def rec(x0,y0,x1,y1,depth):
        if should_split(x0,y0,x1,y1,depth):
            mx=(x0+x1)//2; my=(y0+y1)//2
            if mx==x0 or mx==x1 or my==y0 or my==y1:
                leaves.append((x0,y0,x1,y1)); return
            rec(x0,y0,mx,my,depth+1)
            rec(mx,y0,x1,my,depth+1)
            rec(x0,my,mx,y1,depth+1)
            rec(mx,my,x1,y1,depth+1)
        else:
            leaves.append((x0,y0,x1,y1))
    rec(0,0,w,h,0)
    return leaves

def sample_triangle_color(rgb_u8: np.ndarray, tri: np.ndarray) -> np.ndarray:
    h,w,_ = rgb_u8.shape
    ctr = tri_centroid(tri)
    pts = []
    for t in [0.55, 0.8]:
        for v in tri:
            p = ctr*(1-t) + v*t
            x = int(np.clip(round(p[0]), 0, w-1))
            y = int(np.clip(round(p[1]), 0, h-1))
            pts.append(rgb_u8[y,x].astype(np.float32))
    x = int(np.clip(round(ctr[0]), 0, w-1))
    y = int(np.clip(round(ctr[1]), 0, h-1))
    pts.append(rgb_u8[y,x].astype(np.float32))
    c = np.mean(np.stack(pts,0),0)/255.0
    return np.clip(c,0.0,1.0).astype(np.float32)

def build_graph_from_image(img: Image.Image, qp: QuadParams, seed: int) -> Tuple[TriangleGraph, np.ndarray]:
    img = ImageOps.exif_transpose(img).convert("RGB")
    rgb_u8 = np.array(img, dtype=np.uint8)
    h,w,_ = rgb_u8.shape
    edge = sobel_edge_strength(rgb_u8)
    leaves = build_quadtree_leaves(rgb_u8, edge, qp)
    rng = np.random.default_rng(int(seed))
    tris = []
    for r in leaves:
        tris.extend(rect_to_tris(r, qp.diag_mode, rng))
    tris = np.stack(tris,0).astype(np.float32)
    edges = build_edges_from_tris(tris)
    N = tris.shape[0]
    cols = np.zeros((N,3), np.float32)
    es = np.zeros((N,), np.float32)
    for i in range(N):
        c = sample_triangle_color(rgb_u8, tris[i])
        cols[i] = srgb_to_lin(c)
        ctr = tri_centroid(tris[i])
        cx = int(np.clip(round(ctr[0]), 0, w-1))
        cy = int(np.clip(round(ctr[1]), 0, h-1))
        es[i] = float(edge[cy,cx])
    return TriangleGraph(width=w, height=h, tris=tris, edges=edges, edge_strength=es), cols

# ------------------ Preference Training (A/B) ------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))

def edge_deltas(g: TriangleGraph, colors_lin: np.ndarray) -> np.ndarray:
    if g.edges.shape[0] == 0:
        return np.zeros((0,3), dtype=np.float32)
    i = g.edges[:,0].astype(np.int64)
    j = g.edges[:,1].astype(np.int64)
    return (colors_lin[i] - colors_lin[j]).astype(np.float32)

def compute_quality_features(g: TriangleGraph, colors_lin: np.ndarray) -> np.ndarray:
    d = edge_deltas(g, colors_lin)
    if d.shape[0] == 0:
        var = np.var(colors_lin, axis=0)
        mean = np.mean(colors_lin, axis=0)
        sat = float(np.mean(np.std(lin_to_srgb(colors_lin), axis=1)))
        return np.array([
            float(np.mean(var)), float(np.max(var)), float(np.mean(mean)), sat,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

    dist = np.sqrt(np.sum(d*d, axis=1) + 1e-12)
    es = np.clip(g.edge_strength.astype(np.float32), 0.0, 1.0)
    i = g.edges[:,0].astype(np.int64); j = g.edges[:,1].astype(np.int64)
    s = 0.5*(es[i] + es[j])

    s_mean = float(np.mean(s)); d_mean = float(np.mean(dist))
    cov = float(np.mean((s - s_mean)*(dist - d_mean)))
    s_var = float(np.mean((s - s_mean)**2)) + 1e-8
    d_var = float(np.mean((dist - d_mean)**2)) + 1e-8
    corr = cov / math.sqrt(s_var*d_var)

    smooth_penalty = float(np.mean((1.0 - s) * dist))
    edge_miss_penalty = float(np.mean(s * (1.0 / (dist + 1e-3))))

    var_rgb = np.var(colors_lin, axis=0)
    mean_rgb = np.mean(colors_lin, axis=0)
    max_var = float(np.max(var_rgb))
    mean_var = float(np.mean(var_rgb))
    mean_brightness = float(np.mean(mean_rgb))

    sr = lin_to_srgb(np.clip(colors_lin,0.0,1.0))
    sat = float(np.mean(np.std(sr, axis=1)))

    bins = 8
    q = np.clip((sr*(bins-1)).astype(np.int32), 0, bins-1)
    key = q[:,0]*(bins*bins) + q[:,1]*bins + q[:,2]
    hist = np.bincount(key, minlength=bins**3).astype(np.float32)
    hist = hist/(np.sum(hist)+1e-8)
    ent = float(-np.sum(hist*np.log(hist+1e-8)))

    tv = float(np.mean(dist))
    balance = float(np.std(var_rgb))

    feats = np.array([
        mean_var, max_var, mean_brightness, sat, ent, tv, corr,
        smooth_penalty, edge_miss_penalty, balance,
        float(np.median(dist)), float(np.quantile(dist, 0.90))
    ], dtype=np.float32)
    return feats

def ensure_pref(model: Dict) -> None:
    model.setdefault("pref", {"w":[0.0]*12,"feat_mean":[0.0]*12,"feat_std":[1.0]*12,"lr":0.15,"l2":0.01,"n":0})
    model.setdefault("pref_pairs", [])
    model.setdefault("training", {"tune_rate":0.05,"max_history":4000})

def pref_score(model: Dict, feats: np.ndarray) -> float:
    p = model["pref"]
    w = np.array(p["w"], dtype=np.float32)
    mu = np.array(p["feat_mean"], dtype=np.float32)
    sd = np.array(p["feat_std"], dtype=np.float32) + 1e-6
    x = (feats - mu)/sd
    return float(np.dot(w, x))

def update_pref_stats(model: Dict, feats_all: np.ndarray) -> None:
    if feats_all.shape[0] < 4:
        return
    model["pref"]["feat_mean"] = np.mean(feats_all, axis=0).astype(np.float32).tolist()
    model["pref"]["feat_std"] = (np.std(feats_all, axis=0) + 1e-6).astype(np.float32).tolist()

def train_pref(model: Dict, steps: int = 200) -> None:
    pairs = model.get("pref_pairs", [])
    if len(pairs) < 2:
        model["pref"]["n"] = len(pairs); return
    A = np.stack([np.array(p["fa"], dtype=np.float32) for p in pairs], axis=0)
    B = np.stack([np.array(p["fb"], dtype=np.float32) for p in pairs], axis=0)
    y = np.array([float(p["y"]) for p in pairs], dtype=np.float32)

    feats_all = np.concatenate([A,B], axis=0)
    update_pref_stats(model, feats_all)

    mu = np.array(model["pref"]["feat_mean"], dtype=np.float32)
    sd = np.array(model["pref"]["feat_std"], dtype=np.float32) + 1e-6
    X = (A - mu[None,:])/sd[None,:] - (B - mu[None,:])/sd[None,:]

    w = np.array(model["pref"]["w"], dtype=np.float32)
    lr = float(model["pref"].get("lr", 0.15))
    l2 = float(model["pref"].get("l2", 0.01))
    n = X.shape[0]
    for _ in range(int(steps)):
        p = sigmoid(X @ w)
        grad = (X.T @ (p - y)) / max(1.0, float(n)) + l2*w
        w = w - lr*grad.astype(np.float32)

    model["pref"]["w"] = w.astype(np.float32).tolist()
    model["pref"]["n"] = len(pairs)

def maybe_autotune_defaults(model: Dict, chosen_is_a: bool, params_a: Dict, params_b: Dict) -> None:
    """
    Optional: after a preference, nudge defaults toward winning params.
    This makes the generator improve even if you never use scoring.
    """
    if not model.get("defaults", {}).get("auto_tune_defaults", True):
        return
    tr = model.get("training", {})
    rate = float(tr.get("tune_rate", 0.05))
    win = params_a if chosen_is_a else params_b
    lose = params_b if chosen_is_a else params_a

    # only tune a curated subset (keeps stable)
    keys = ["neighbor_weight","prompt_weight","edge_preserve","temperature","multi_scale","quantize_strength","micro_texture","contrast_boost"]
    for k in keys:
        if k in win and k in lose and k in model["defaults"]:
            wv = float(win[k]); lv = float(lose[k])
            dv = wv - lv
            cur = float(model["defaults"][k])
            model["defaults"][k] = float(np.clip(cur + rate*dv, 0.0, 2.5))

def save_model(model: Dict) -> None:
    with open("model.json","w",encoding="utf-8") as f:
        json.dump(model, f, indent=2)

# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="TriangleGraph Text-to-Image v2", page_icon="🔺", layout="wide")
st.title("🔺 TriangleGraph Text-to-Image — v2 (No Torch)")
st.caption("Stronger prompt conditioning + stained-glass look + training mode.")

@st.cache_data(show_spinner=False)
def load_model() -> Dict:
    with open("model.json","r",encoding="utf-8") as f:
        return json.load(f)

MODEL = load_model()
ensure_pref(MODEL)

tab1, tab2, tab3 = st.tabs(["Text → Image", "Image → Triangle Mosaic", "Train (A/B)"])

# -------------------- Text → Image --------------------
with tab1:
    st.subheader("Text → Image (Generates ONE image per click)")
    prompt = st.text_input("Prompt", value="ocean sunset, stained glass, light from left")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        width = st.select_slider("Width", options=[384, 512, 640, 768, 896, 1024], value=768)
    with colB:
        height = st.select_slider("Height", options=[384, 512, 640, 768, 896, 1024], value=768)
    with colC:
        complexity = st.slider("Complexity (more triangles)", 1, 10, 6, 1)
    with colD:
        seed_default = stable_hash_int(prompt) % 10_000_000
        seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=int(seed_default), step=1)

    style = detect_style(MODEL, prompt)
    style_cfg = MODEL.get("styles", {}).get(style, {})
    st.write(f"Detected style: **{style}**")

    st.markdown("### Style controls")
    d = MODEL.get("defaults", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        steps = st.slider("Steps", 10, 140, int(d.get("steps", 40)), 1)
    with c2:
        neighbor_w = st.slider("Neighbor smooth", 0.10, 0.92, float(d.get("neighbor_weight", 0.62)), 0.01)
    with c3:
        prompt_w = st.slider("Prompt pull", 0.00, 0.65, float(d.get("prompt_weight", 0.28)), 0.01)
    with c4:
        edge_preserve = st.slider("Edge preserve", 0.00, 0.75, float(d.get("edge_preserve", 0.22)), 0.01)
    with c5:
        temperature = st.slider("Randomness", 0.10, 1.50, float(d.get("temperature", 0.85)), 0.05)

    st.markdown("### Advanced (quality boosters)")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        multi_scale = st.slider("Multi-scale structure", 0.0, 1.0, float(d.get("multi_scale", 0.55)), 0.01)
    with a2:
        quantize_strength = st.slider("Palette quantize", 0.0, 0.85, float(d.get("quantize_strength", 0.45)), 0.01)
    with a3:
        micro_texture = st.slider("Glass grain", 0.0, 0.55, float(d.get("micro_texture", 0.22)), 0.01)
    with a4:
        contrast_boost = st.slider("Edge contrast", 0.0, 0.25, float(d.get("contrast_boost", 0.08)), 0.01)

    # blend in style defaults subtly
    quantize_strength = float(np.clip(0.70*quantize_strength + 0.30*float(style_cfg.get("quantize_strength", quantize_strength)), 0.0, 0.85))
    micro_texture = float(np.clip(0.70*micro_texture + 0.30*float(style_cfg.get("micro_texture", micro_texture)), 0.0, 0.55))
    contrast_boost = float(np.clip(0.70*contrast_boost + 0.30*float(style_cfg.get("contrast_boost", contrast_boost)), 0.0, 0.25))
    lead_line_strength = float(style_cfg.get("lead_line_strength", 0.25))

    st.markdown("### Render controls")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        gap = st.slider("Whitespace gap (px)", 0.0, 28.0, 10.0, 0.5)
    with r2:
        bg = st.radio("Background", ["White","Black"], horizontal=True)
    with r3:
        outline = st.checkbox("Outlines", value=False)
    with r4:
        outline_px = st.slider("Outline width", 1, 6, 1, 1) if outline else 1
        outline_alpha = st.slider("Outline opacity", 0.05, 1.0, 0.25, 0.01) if outline else 0.25

    diag_mode = st.selectbox("Diagonal mode", ["Random","Alternate","TL-BR","TR-BL"], index=0)

    # graph params derived from complexity
    min_cell = int(max(10, 70 - 5*complexity))
    max_depth = int(min(9, 3 + complexity//2))
    split_prob = float(np.clip(0.35 + 0.06*complexity, 0.35, 0.92))

    gen = st.button("Generate image", type="primary", use_container_width=True)

    if gen:
        t0 = time.time()
        with st.spinner("Generating (one image)…"):
            g = build_procedural_graph(int(width), int(height), ProcParams(min_cell=min_cell, max_depth=max_depth, split_prob=split_prob, diag_mode=diag_mode), seed=int(seed))
            colors = generate_colors(
                MODEL, g, prompt=prompt, seed=int(seed), steps=int(steps),
                neighbor_w=float(neighbor_w), prompt_w=float(prompt_w),
                edge_preserve=float(edge_preserve), temperature=float(temperature),
                multi_scale=float(multi_scale), quantize_strength=float(quantize_strength),
                micro_texture=float(micro_texture), contrast_boost=float(contrast_boost),
            )
            img = render_mosaic(g, colors, gap_px=float(gap), background=bg, outline=outline,
                                outline_px=int(outline_px), outline_alpha=float(outline_alpha),
                                lead_line_strength=lead_line_strength)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

        st.image(png_bytes, use_container_width=True)
        st.download_button("Download PNG", data=png_bytes, file_name="trianglegraph_text2image.png", mime="image/png", use_container_width=True)
        st.caption(f"Generated in {time.time()-t0:.2f}s • triangles={int(g.tris.shape[0])} • edges={int(g.edges.shape[0])}")

        with st.expander("Generation details", expanded=False):
            st.write({
                "prompt": prompt, "style": style,
                "width": width, "height": height, "seed": int(seed),
                "min_cell": min_cell, "max_depth": max_depth, "split_prob": split_prob, "diag_mode": diag_mode,
                "steps": int(steps),
                "neighbor_w": float(neighbor_w), "prompt_w": float(prompt_w),
                "edge_preserve": float(edge_preserve), "temperature": float(temperature),
                "multi_scale": float(multi_scale), "quantize_strength": float(quantize_strength),
                "micro_texture": float(micro_texture), "contrast_boost": float(contrast_boost),
            })

# -------------------- Image → Triangle Mosaic --------------------
with tab2:
    st.subheader("Image → Triangle Mosaic")
    up = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp"])
    if up is not None:
        img = Image.open(io.BytesIO(up.getvalue()))
        img = ImageOps.exif_transpose(img).convert("RGB")
        max_side = st.slider("Max side (px)", 256, 1600, 900, 32)
        img = resize_keep_aspect(img, int(max_side))

        c1, c2, c3, c4 = st.columns(4)
        with c1: min_cell = st.slider("Min cell (px)", 6, 80, 18, 1)
        with c2: max_depth = st.slider("Max depth", 2, 10, 7, 1)
        with c3: var_thresh = st.slider("Var thresh", 0.05, 1.0, 0.55, 0.01)
        with c4: edge_thresh = st.slider("Edge thresh", 0.05, 1.0, 0.45, 0.01)

        edge_weight = st.slider("Edge weighting", 0.0, 1.0, 0.60, 0.01)
        diag_mode = st.selectbox("Diagonal mode", ["Random","Alternate","TL-BR","TR-BL"], index=0, key="img_diag")
        seed = st.number_input("Seed", 0, 10_000_000, 42, 1, key="img_seed")

        gap = st.slider("Whitespace gap (px)", 0.0, 28.0, 8.0, 0.5, key="img_gap")
        bg = st.radio("Background", ["White","Black"], horizontal=True, key="img_bg")
        outline = st.checkbox("Outlines", value=False, key="img_out")
        outline_px = st.slider("Outline width", 1, 6, 1, 1, key="img_opx") if outline else 1
        outline_alpha = st.slider("Outline opacity", 0.05, 1.0, 0.25, 0.01, key="img_oal") if outline else 0.25

        run = st.button("Convert image", type="primary", use_container_width=True)
        if run:
            with st.spinner("Converting…"):
                qp = QuadParams(min_cell=int(min_cell), max_depth=int(max_depth),
                                var_thresh=float(var_thresh), edge_thresh=float(edge_thresh),
                                edge_weight=float(edge_weight), diag_mode=diag_mode)
                g, cols = build_graph_from_image(img, qp, seed=int(seed))
                out = render_mosaic(g, cols, gap_px=float(gap), background=bg, outline=outline,
                                    outline_px=int(outline_px), outline_alpha=float(outline_alpha),
                                    lead_line_strength=0.0)
                buf = io.BytesIO()
                out.save(buf, format="PNG")
                png = buf.getvalue()

            st.image(png, use_container_width=True)
            st.download_button("Download PNG", data=png, file_name="trianglegraph_from_image.png", mime="image/png", use_container_width=True)

# -------------------- Train (A/B) --------------------
with tab3:
    st.subheader("Train (A/B Preferences) — improves your generator over time")
    st.write(
        "This tab intentionally generates **two** candidates (A and B) so you can choose the better one. "
        "It updates a lightweight preference model and can auto-tune defaults in `model.json`.\n\n"
        "If you want training without showing two images at once, ask and I'll convert this into a 2-click sequential trainer."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: t_width = st.select_slider("Width", options=[384, 512, 640, 768, 896, 1024], value=768, key="t_w")
    with c2: t_height = st.select_slider("Height", options=[384, 512, 640, 768, 896, 1024], value=768, key="t_h")
    with c3: t_complexity = st.slider("Complexity", 1, 10, 6, 1, key="t_c")
    with c4: t_seed = st.number_input("Base seed", min_value=0, max_value=10_000_000, value=9001, step=1, key="t_seed")

    t_prompt = st.text_input("Prompt", value="neon cyber stained glass, light from top-right", key="t_prompt")

    st.markdown("### A/B generation knobs")
    d = MODEL.get("defaults", {})
    s1, s2, s3, s4 = st.columns(4)
    with s1: t_steps = st.slider("Steps", 10, 140, int(d.get("steps", 40)), 1, key="t_steps")
    with s2: t_temp = st.slider("Randomness", 0.10, 1.50, float(d.get("temperature", 0.85)), 0.05, key="t_tmp")
    with s3: spread = st.slider("Param exploration", 0.00, 0.30, 0.10, 0.01, key="t_sp")
    with s4:
        train_steps = st.slider("Pref train steps per vote", 50, 600, 200, 25, key="t_tr")

    diag_mode = st.selectbox("Diagonal mode", ["Random","Alternate","TL-BR","TR-BL"], index=0, key="t_diag")

    r1, r2, r3 = st.columns(3)
    with r1: t_gap = st.slider("Whitespace gap (px)", 0.0, 28.0, 10.0, 0.5, key="t_gap")
    with r2: t_bg = st.radio("Background", ["White","Black"], horizontal=True, key="t_bg")
    with r3:
        t_outline = st.checkbox("Outlines", value=False, key="t_out")
        t_opx = st.slider("Outline width", 1, 6, 1, 1, key="t_opx") if t_outline else 1
        t_oal = st.slider("Outline opacity", 0.05, 1.0, 0.25, 0.01, key="t_oal") if t_outline else 0.25

    # graph params from complexity
    min_cell = int(max(10, 70 - 5*t_complexity))
    max_depth = int(min(9, 3 + t_complexity//2))
    split_prob = float(np.clip(0.35 + 0.06*t_complexity, 0.35, 0.92))

    gen_pair = st.button("Generate A/B pair", type="primary", use_container_width=True)

    def jitter_param(v: float, lo: float, hi: float, rng: np.random.Generator) -> float:
        return float(np.clip(v + rng.normal(0.0, spread), lo, hi))

    if gen_pair:
        rng = np.random.default_rng(int(t_seed))
        with st.spinner("Generating A and B…"):
            g = build_procedural_graph(int(t_width), int(t_height),
                                       ProcParams(min_cell=min_cell, max_depth=max_depth, split_prob=split_prob, diag_mode=diag_mode),
                                       seed=int(t_seed))

            style = detect_style(MODEL, t_prompt)
            style_cfg = MODEL.get("styles", {}).get(style, {})
            lead = float(style_cfg.get("lead_line_strength", 0.25))

            # Candidate A (baseline defaults with small jitter)
            params_a = {
                "neighbor_weight": jitter_param(float(d.get("neighbor_weight",0.62)), 0.10, 0.92, rng),
                "prompt_weight": jitter_param(float(d.get("prompt_weight",0.28)), 0.00, 0.65, rng),
                "edge_preserve": jitter_param(float(d.get("edge_preserve",0.22)), 0.00, 0.75, rng),
                "temperature": float(t_temp),
                "multi_scale": jitter_param(float(d.get("multi_scale",0.55)), 0.00, 1.00, rng),
                "quantize_strength": jitter_param(float(d.get("quantize_strength",0.45)), 0.00, 0.85, rng),
                "micro_texture": jitter_param(float(d.get("micro_texture",0.22)), 0.00, 0.55, rng),
                "contrast_boost": jitter_param(float(d.get("contrast_boost",0.08)), 0.00, 0.25, rng),
            }
            colors_a = generate_colors(MODEL, g, prompt=t_prompt, seed=int(t_seed)+7, steps=int(t_steps),
                                       neighbor_w=params_a["neighbor_weight"], prompt_w=params_a["prompt_weight"],
                                       edge_preserve=params_a["edge_preserve"], temperature=params_a["temperature"],
                                       multi_scale=params_a["multi_scale"], quantize_strength=params_a["quantize_strength"],
                                       micro_texture=params_a["micro_texture"], contrast_boost=params_a["contrast_boost"])
            img_a = render_mosaic(g, colors_a, gap_px=float(t_gap), background=t_bg, outline=t_outline,
                                  outline_px=int(t_opx), outline_alpha=float(t_oal), lead_line_strength=lead)
            fa = compute_quality_features(g, colors_a)

            # Candidate B (different seed + jitter)
            params_b = {
                "neighbor_weight": jitter_param(float(d.get("neighbor_weight",0.62)), 0.10, 0.92, rng),
                "prompt_weight": jitter_param(float(d.get("prompt_weight",0.28)), 0.00, 0.65, rng),
                "edge_preserve": jitter_param(float(d.get("edge_preserve",0.22)), 0.00, 0.75, rng),
                "temperature": float(t_temp),
                "multi_scale": jitter_param(float(d.get("multi_scale",0.55)), 0.00, 1.00, rng),
                "quantize_strength": jitter_param(float(d.get("quantize_strength",0.45)), 0.00, 0.85, rng),
                "micro_texture": jitter_param(float(d.get("micro_texture",0.22)), 0.00, 0.55, rng),
                "contrast_boost": jitter_param(float(d.get("contrast_boost",0.08)), 0.00, 0.25, rng),
            }
            colors_b = generate_colors(MODEL, g, prompt=t_prompt, seed=int(t_seed)+1007, steps=int(t_steps),
                                       neighbor_w=params_b["neighbor_weight"], prompt_w=params_b["prompt_weight"],
                                       edge_preserve=params_b["edge_preserve"], temperature=params_b["temperature"],
                                       multi_scale=params_b["multi_scale"], quantize_strength=params_b["quantize_strength"],
                                       micro_texture=params_b["micro_texture"], contrast_boost=params_b["contrast_boost"])
            img_b = render_mosaic(g, colors_b, gap_px=float(t_gap), background=t_bg, outline=t_outline,
                                  outline_px=int(t_opx), outline_alpha=float(t_oal), lead_line_strength=lead)
            fb = compute_quality_features(g, colors_b)

            buf = io.BytesIO(); img_a.save(buf, format="PNG"); a_png = buf.getvalue()
            buf = io.BytesIO(); img_b.save(buf, format="PNG"); b_png = buf.getvalue()

        st.session_state["ab_pair"] = {
            "a_png": a_png, "b_png": b_png,
            "fa": fa.tolist(), "fb": fb.tolist(),
            "params_a": params_a, "params_b": params_b
        }
        st.success("Pair generated. Choose A or B below.")

    pair = st.session_state.get("ab_pair")
    if pair is not None:
        a_sc = pref_score(MODEL, np.array(pair["fa"], dtype=np.float32))
        b_sc = pref_score(MODEL, np.array(pair["fb"], dtype=np.float32))

        cA, cB = st.columns(2, gap="large")
        with cA:
            st.markdown(f"## A (score {a_sc:.3f})")
            st.image(pair["a_png"], use_container_width=True)
            st.caption(pair["params_a"])
        with cB:
            st.markdown(f"## B (score {b_sc:.3f})")
            st.image(pair["b_png"], use_container_width=True)
            st.caption(pair["params_b"])

        b1, b2, b3 = st.columns(3)
        choose_a = b1.button("✅ A is better", use_container_width=True)
        choose_b = b2.button("✅ B is better", use_container_width=True)
        clear = b3.button("↩ Clear", use_container_width=True)

        if choose_a or choose_b:
            chosen_is_a = bool(choose_a)
            entry = {"fa": pair["fa"], "fb": pair["fb"], "y": 1.0 if chosen_is_a else 0.0}
            MODEL["pref_pairs"].append(entry)

            # keep model.json from growing forever
            max_hist = int(MODEL.get("training", {}).get("max_history", 4000))
            if len(MODEL["pref_pairs"]) > max_hist:
                MODEL["pref_pairs"] = MODEL["pref_pairs"][-max_hist:]

            with st.spinner("Training preference model & saving…"):
                train_pref(MODEL, steps=int(train_steps))
                maybe_autotune_defaults(MODEL, chosen_is_a, pair["params_a"], pair["params_b"])
                save_model(MODEL)
            st.success(f"Saved. Comparisons: {len(MODEL['pref_pairs'])} • pref n={MODEL['pref']['n']}")
            st.session_state.pop("ab_pair", None)

        if clear:
            st.session_state.pop("ab_pair", None)

    st.divider()
    st.markdown("### Save / Export")
    colS1, colS2 = st.columns(2)
    with colS1:
        if st.button("Save model.json now", use_container_width=True):
            save_model(MODEL)
            st.success("Saved model.json")
    with colS2:
        model_bytes = json.dumps(MODEL, indent=2).encode("utf-8")
        st.download_button("Download model.json", data=model_bytes, file_name="model.json", mime="application/json", use_container_width=True)

    st.markdown("### Status")
    st.write({"comparisons": len(MODEL.get("pref_pairs", [])), "defaults": MODEL.get("defaults", {}), "pref": MODEL.get("pref", {})})

with st.expander("Troubleshooting", expanded=False):
    st.write(
        "- If generation feels too 'washed', increase **Edge contrast** and reduce **Neighbor smooth**.\n"
        "- If it looks too noisy, reduce **Randomness** and increase **Steps**.\n"
        "- For stained glass: raise **Palette quantize** and **Glass grain**, and enable outlines.\n"
        "- If you want the generator to improve automatically, do ~25-100 A/B votes and keep Auto-tune enabled.\n"
    )
