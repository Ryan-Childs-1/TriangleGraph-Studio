import io, json, math, hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

# ============================================================
# TriangleGraph Text-to-Image (No Torch)
# - Generates ONE image per click (to avoid crashes)
# - Text prompt -> palette + seed + directional light
# - Procedural triangle graph -> lightweight generative model
# ============================================================

# ------------------ Utility ------------------

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def hex_to_rgb01(h: str) -> np.ndarray:
    h = h.strip().lstrip("#")
    if len(h) != 6:
        return np.array([0.5, 0.5, 0.5], dtype=np.float32)
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return np.array([r, g, b], dtype=np.float32) / 255.0

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

def stable_hash_int(s: str, mod: int = 2**31-1) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % mod

def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    s = max_side / max(w, h)
    return img.resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS)

# ------------------ Geometry ------------------

def tri_centroid(t: np.ndarray) -> np.ndarray:
    return np.mean(t.astype(np.float32), axis=0)

def tri_area(t: np.ndarray) -> float:
    a, b, c = t.astype(np.float32)
    return float(abs(np.cross(b - a, c - a)) * 0.5)

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

# ------------------ Graph ------------------

@dataclass
class TriangleGraph:
    width: int
    height: int
    tris: np.ndarray      # (N,3,2) float32
    edges: np.ndarray     # (E,2) int64
    edge_strength: np.ndarray  # (N,) float32  synthetic edge strength per triangle

def build_edges_from_tris(tris: np.ndarray) -> np.ndarray:
    local_edges = [(0,1),(1,2),(2,0)]
    edge_dict: Dict[Tuple[int,int,int,int], Tuple[int,int]] = {}
    pairs = []
    for ti in range(tris.shape[0]):
        t = tris[ti]
        for le,(a,b) in enumerate(local_edges):
            key = _edge_key(t[a], t[b])
            if key not in edge_dict:
                edge_dict[key] = (ti, le)
            else:
                tj,_ = edge_dict[key]
                if tj != ti:
                    pairs.append((tj, ti))
                del edge_dict[key]
    if not pairs:
        return np.zeros((0,2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)

def build_neighbor_lists(edges: np.ndarray, N: int) -> List[List[int]]:
    nbr = [[] for _ in range(N)]
    for i,j in edges.astype(np.int64):
        nbr[int(i)].append(int(j))
        nbr[int(j)].append(int(i))
    return nbr

# ------------------ Procedural triangulation ------------------
# A quadtree-like split controlled by "complexity" and seed.
# This avoids needing an input image for geometry.

@dataclass
class ProcParams:
    min_cell: int
    max_depth: int
    split_prob: float
    diag_mode: str  # Random / Alternate / TL-BR / TR-BL

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

def build_procedural_graph(width: int, height: int, params: ProcParams, seed: int) -> TriangleGraph:
    rng = np.random.default_rng(int(seed))
    leaves: List[Tuple[int,int,int,int]] = []

    def rec(x0,y0,x1,y1,depth):
        rw, rh = x1-x0, y1-y0
        if rw <= params.min_cell or rh <= params.min_cell or depth >= params.max_depth:
            leaves.append((x0,y0,x1,y1)); return
        # split decision: probability + a little spatial noise
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
    N = tris.shape[0]
    # synthetic edge strength: higher near "guiding lines" to create structure
    ctrs = np.mean(tris, axis=1)  # (N,2)
    x = ctrs[:,0] / max(1.0,width)
    y = ctrs[:,1] / max(1.0,height)
    # create 2–3 sine-based bands
    band1 = np.abs(np.sin(2*math.pi*(1.5*x + 0.7*y + 0.13)))
    band2 = np.abs(np.sin(2*math.pi*(0.6*x - 1.2*y + 0.41)))
    edge_strength = (0.55*band1 + 0.45*band2).astype(np.float32)
    edge_strength = np.clip(edge_strength, 0.0, 1.0)

    return TriangleGraph(width=width, height=height, tris=tris, edges=edges, edge_strength=edge_strength)

# ------------------ Rendering ------------------

def render_mosaic(g: TriangleGraph, colors_lin: np.ndarray, gap_px: float, background: str, outline: bool, outline_px: int, outline_alpha: float) -> Image.Image:
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

    for i in range(g.tris.shape[0]):
        t = g.tris[i]
        ts = shrink_triangle(t, float(gap_px))
        poly = [(float(p[0]), float(p[1])) for p in ts]
        fill = (int(cu8[i,0]), int(cu8[i,1]), int(cu8[i,2]), 255)
        draw.polygon(poly, fill=fill)
        if outline_rgba is not None and outline_px>0:
            draw.line(poly+[poly[0]], fill=outline_rgba, width=int(outline_px), joint="curve")
    return canvas

# ------------------ Image->Graph (optional) ------------------

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
    hi = np.quantile(mag, 0.995) + 1e-8
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

    # triangle colors from image (linear)
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

# ------------------ Text->Palette + Light ------------------

def parse_light(prompt: str) -> Tuple[float,float]:
    p = prompt.lower()
    # default: top-left
    dx, dy = -0.7, -0.7
    if "light from left" in p or "from left" in p:
        dx, dy = -1.0, 0.0
    if "light from right" in p or "from right" in p:
        dx, dy = 1.0, 0.0
    if "light from above" in p or "from above" in p or "from top" in p:
        dx, dy = 0.0, -1.0
    if "light from below" in p or "from below" in p or "from bottom" in p:
        dx, dy = 0.0, 1.0
    if "top-right" in p:
        dx, dy = 0.7, -0.7
    if "bottom-right" in p:
        dx, dy = 0.7, 0.7
    if "bottom-left" in p:
        dx, dy = -0.7, 0.7
    return dx, dy

def pick_palette(model: Dict, prompt: str) -> List[np.ndarray]:
    p = prompt.lower()
    pal = model["palettes"]

    # keyword routing
    if any(k in p for k in ["ocean","sea","water","wave","navy","blue"]): key="ocean"
    elif any(k in p for k in ["forest","jungle","green","moss","tree"]): key="forest"
    elif any(k in p for k in ["neon","cyber","synth","vapor","glow"]): key="neon"
    elif any(k in p for k in ["sunset","dawn","golden","warm","desert"]): key="sunset"
    elif any(k in p for k in ["black and white","monochrome","mono","grayscale"]): key="mono"
    elif any(k in p for k in ["candy","pastel","sweet"]): key="candy"
    else:
        # pick based on hash
        keys = list(pal.keys())
        key = keys[stable_hash_int(prompt) % len(keys)]

    return [srgb_to_lin(hex_to_rgb01(hx)) for hx in pal[key]]

# ------------------ Generation Model ------------------
# Lightweight iterative graph denoising, edge-aware + prompt palette bias.
# Produces ONE final colors array.

def build_neighbor_lists(edges: np.ndarray, N: int) -> List[List[int]]:
    nbr = [[] for _ in range(N)]
    for i,j in edges.astype(np.int64):
        nbr[int(i)].append(int(j)); nbr[int(j)].append(int(i))
    return nbr

def prompt_field(g: TriangleGraph, palette: List[np.ndarray], prompt: str, seed: int) -> np.ndarray:
    """
    Create a smooth prompt-conditioned color field across triangles.
    Uses centroid position + hashed prompt direction + palette interpolation.
    """
    rng = np.random.default_rng(int(seed))
    ctr = np.mean(g.tris, axis=1)  # (N,2)
    x = ctr[:,0] / max(1.0,g.width)
    y = ctr[:,1] / max(1.0,g.height)

    dx, dy = parse_light(prompt)
    # projection for "lighting direction"
    proj = dx*(x-0.5) + dy*(y-0.5)
    proj = (proj - proj.min()) / (proj.ptp() + 1e-8)

    # a second axis for variety
    proj2 = (x + 0.6*y + 0.17*np.sin(2*math.pi*(x-y))) % 1.0

    # mix palette stops
    K = len(palette)
    t = 0.65*proj + 0.35*proj2
    t = np.clip(t, 0.0, 0.999999)
    pos = t*(K-1)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.clip(i0+1, 0, K-1)
    a = (pos - i0.astype(np.float32)).astype(np.float32)

    base = (1-a)[:,None]*np.stack([palette[i] for i in i0],0) + a[:,None]*np.stack([palette[i] for i in i1],0)

    # small prompt texture
    tex = rng.normal(0.0, 0.03, size=base.shape).astype(np.float32)
    return np.clip(base + tex, 0.0, 1.0).astype(np.float32)

def generate_colors(model: Dict, g: TriangleGraph, prompt: str, seed: int, steps: int, neighbor_w: float, prompt_w: float, edge_preserve: float, temperature: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    N = g.tris.shape[0]
    nbr = build_neighbor_lists(g.edges, N)

    palette = pick_palette(model, prompt)
    pf = prompt_field(g, palette, prompt, seed=seed+1234)

    # initialize with palette-biased noise
    x = np.clip(pf + rng.normal(0.0, 0.25*temperature, size=(N,3)).astype(np.float32), 0.0, 1.0)

    es = np.clip(g.edge_strength.astype(np.float32), 0.0, 1.0)  # (N,)
    # edge-preserve mask: where edges strong, resist smoothing
    resist = edge_preserve * es  # 0..edge_preserve

    for _ in range(int(steps)):
        x_new = x.copy()
        # neighbor mean
        for i in range(N):
            ns = nbr[i]
            if ns:
                m = np.mean(x[ns], axis=0)
            else:
                m = x[i]
            # smooth more where edges are weak
            w_smooth = neighbor_w * (1.0 - resist[i])
            # pull toward prompt field always a bit
            x_new[i] = (1.0 - w_smooth - prompt_w) * x[i] + w_smooth * m + prompt_w * pf[i]

        # mild contrast boost aligned with edges
        # encourages separation along structural edges
        if g.edges.shape[0] > 0:
            i = g.edges[:,0].astype(np.int64); j = g.edges[:,1].astype(np.int64)
            d = x_new[i] - x_new[j]
            mag = np.sqrt(np.sum(d*d, axis=1) + 1e-12)
            # push apart along strong edges
            push = (0.12 * (g.edge_strength[i] + g.edge_strength[j]) / 2.0)[:,None]
            x_new[i] = np.clip(x_new[i] + push * d, 0.0, 1.0)
            x_new[j] = np.clip(x_new[j] - push * d, 0.0, 1.0)

        # anneal small noise
        x_new = np.clip(x_new + rng.normal(0.0, 0.02*temperature, size=x_new.shape).astype(np.float32), 0.0, 1.0)
        x = x_new

    return np.clip(x, 0.0, 1.0).astype(np.float32)

# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="TriangleGraph Text-to-Image", page_icon="🔺", layout="wide")
st.title("🔺 TriangleGraph Text-to-Image (No Torch)")
st.caption("Generates ONE triangle-mosaic image per click from a text prompt. Streamlit-friendly.")

# Load model.json from local folder
@st.cache_data(show_spinner=False)
def load_model() -> Dict:
    with open("model.json","r",encoding="utf-8") as f:
        return json.load(f)

MODEL = load_model()

tab1, tab2 = st.tabs(["Text → Image", "Image → Triangle Mosaic"])

# -------------------- Text → Image --------------------
with tab1:
    st.subheader("Text → Image")
    prompt = st.text_input("Prompt", value="ocean sunset, light from left, stained glass")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        width = st.select_slider("Width", options=[384, 512, 640, 768, 896, 1024], value=768)
    with colB:
        height = st.select_slider("Height", options=[384, 512, 640, 768, 896, 1024], value=768)
    with colC:
        complexity = st.slider("Complexity (more triangles)", 1, 10, 6, 1)
    with colD:
        seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=stable_hash_int(prompt) % 10_000_000, step=1)

    st.markdown("### Style controls")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        steps = st.slider("Steps", 10, 120, int(MODEL["defaults"]["steps"]), 1)
    with c2:
        neighbor_w = st.slider("Neighbor smooth", 0.10, 0.90, float(MODEL["defaults"]["neighbor_weight"]), 0.01)
    with c3:
        prompt_w = st.slider("Prompt pull", 0.00, 0.60, float(MODEL["defaults"]["prompt_weight"]), 0.01)
    with c4:
        edge_preserve = st.slider("Edge preserve", 0.00, 0.70, float(MODEL["defaults"]["edge_preserve"]), 0.01)
    with c5:
        temperature = st.slider("Randomness", 0.10, 1.40, float(MODEL["defaults"]["temperature"]), 0.05)

    st.markdown("### Render controls")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        gap = st.slider("Whitespace gap (px)", 0.0, 28.0, 10.0, 0.5)
    with r2:
        bg = st.radio("Background", ["White","Black"], horizontal=True)
    with r3:
        outline = st.checkbox("Outlines", value=False)
    with r4:
        outline_px = st.slider("Outline width", 1, 5, 1, 1) if outline else 1
        outline_alpha = st.slider("Outline opacity", 0.05, 1.0, 0.25, 0.01) if outline else 0.25

    # Procedural graph params derived from complexity
    # higher complexity => smaller min_cell and higher depth/split_prob
    min_cell = int(max(10, 70 - 5*complexity))
    max_depth = int(min(9, 3 + complexity//2))
    split_prob = float(np.clip(0.35 + 0.06*complexity, 0.35, 0.92))

    diag_mode = st.selectbox("Diagonal mode", ["Random","Alternate","TL-BR","TR-BL"], index=0)

    gen = st.button("Generate image", type="primary", use_container_width=True)

    if gen:
        # Ensure only ONE image generation per click; avoid storing huge objects in session.
        with st.spinner("Generating (one image)…"):
            g = build_procedural_graph(int(width), int(height), ProcParams(min_cell=min_cell, max_depth=max_depth, split_prob=split_prob, diag_mode=diag_mode), seed=int(seed))
            colors = generate_colors(MODEL, g, prompt=prompt, seed=int(seed), steps=int(steps),
                                     neighbor_w=float(neighbor_w), prompt_w=float(prompt_w),
                                     edge_preserve=float(edge_preserve), temperature=float(temperature))
            img = render_mosaic(g, colors, gap_px=float(gap), background=bg, outline=outline, outline_px=int(outline_px), outline_alpha=float(outline_alpha))

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

        st.image(png_bytes, use_container_width=True)
        st.download_button("Download PNG", data=png_bytes, file_name="trianglegraph_text2image.png", mime="image/png", use_container_width=True)

        with st.expander("Generation details", expanded=False):
            st.write({
                "width": width, "height": height, "seed": int(seed),
                "min_cell": min_cell, "max_depth": max_depth, "split_prob": split_prob,
                "steps": int(steps),
                "neighbor_w": float(neighbor_w), "prompt_w": float(prompt_w),
                "edge_preserve": float(edge_preserve), "temperature": float(temperature),
                "triangles": int(g.tris.shape[0]), "edges": int(g.edges.shape[0])
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
        with c1:
            min_cell = st.slider("Min cell (px)", 6, 80, 18, 1)
        with c2:
            max_depth = st.slider("Max depth", 2, 10, 7, 1)
        with c3:
            var_thresh = st.slider("Var thresh", 0.05, 1.0, 0.55, 0.01)
        with c4:
            edge_thresh = st.slider("Edge thresh", 0.05, 1.0, 0.45, 0.01)

        edge_weight = st.slider("Edge weighting", 0.0, 1.0, 0.60, 0.01)
        diag_mode = st.selectbox("Diagonal mode", ["Random","Alternate","TL-BR","TR-BL"], index=0, key="img_diag")
        seed = st.number_input("Seed", 0, 10_000_000, 42, 1, key="img_seed")

        gap = st.slider("Whitespace gap (px)", 0.0, 28.0, 8.0, 0.5, key="img_gap")
        bg = st.radio("Background", ["White","Black"], horizontal=True, key="img_bg")
        outline = st.checkbox("Outlines", value=False, key="img_out")
        outline_px = st.slider("Outline width", 1, 5, 1, 1, key="img_opx") if outline else 1
        outline_alpha = st.slider("Outline opacity", 0.05, 1.0, 0.25, 0.01, key="img_oal") if outline else 0.25

        run = st.button("Convert image", type="primary", use_container_width=True)
        if run:
            with st.spinner("Converting…"):
                qp = QuadParams(min_cell=int(min_cell), max_depth=int(max_depth),
                                var_thresh=float(var_thresh), edge_thresh=float(edge_thresh),
                                edge_weight=float(edge_weight), diag_mode=diag_mode)
                g, cols = build_graph_from_image(img, qp, seed=int(seed))
                out = render_mosaic(g, cols, gap_px=float(gap), background=bg, outline=outline, outline_px=int(outline_px), outline_alpha=float(outline_alpha))
                buf = io.BytesIO()
                out.save(buf, format="PNG")
                png = buf.getvalue()

            st.image(png, use_container_width=True)
            st.download_button("Download PNG", data=png, file_name="trianglegraph_from_image.png", mime="image/png", use_container_width=True)

with st.expander("Notes / Why this won't crash like before", expanded=False):
    st.write(
        "- This app generates **exactly one** image per click (no best-of-N loops).\n"
        "- It avoids caching large numpy arrays in session state.\n"
        "- It stores only the final PNG bytes for display/download.\n"
        "- The generation model is deterministic by (prompt, seed) and lightweight.\n"
    )
