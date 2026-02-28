# TriangleGraph Text-to-Image (No Torch)

A Streamlit app that generates a **triangle-mosaic image from a text prompt**, and can also convert an uploaded image into a triangle mosaic.
No PyTorch / Torch required. Streamlit Cloud-friendly.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What "text-to-image" means here
This project generates images in a *mosaic / low-poly / stained-glass* aesthetic by:

1) Building a procedural triangle graph (a quadtree-like triangulation).
2) Turning your text prompt into a style seed + palette.
3) Running a lightweight graph-based generative model (iterative denoise/smooth with edge-aware preservation).
4) Rendering triangles with optional whitespace gap and outlines.

This is intentionally **structure-first** generation: it generates triangle color relationships rather than pixels.

## Files
- `app.py` — Streamlit app
- `model.json` — lightweight generation model config (palettes + weights)
- `requirements.txt`
