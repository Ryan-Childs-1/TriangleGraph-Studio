# TriangleGraph Text-to-Image (No Torch) — v2

This is a Streamlit-friendly, no-torch image generator that creates **triangle mosaic / stained-glass** images from text.

### Highlights (v2)
- Much stronger prompt → palette/field mapping (keywords + style tags + lighting direction).
- Multi-scale color field + edge-aware anisotropic smoothing.
- Optional palette-quantization for stained glass look (soft quantization).
- Micro-texture to mimic glass grain (still lightweight).
- Training (A/B): updates preference model AND can auto-tune defaults (optional).
- Generates **one** image per click in the Text→Image tab.

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```
