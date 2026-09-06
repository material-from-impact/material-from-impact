"""
Convert ObjectFolder-Real meshes (model.obj + texture.png) to LIGHTWEIGHT textured
GLB for <model-viewer> on the project page. Meshes are decimated so three viewers
stay smooth. Output: gitpage/assets/models/obj_<id>.glb
"""
import os
import numpy as np
import trimesh
from PIL import Image

SRC = "assets/objf_real"
OUT = "gitpage/assets/models"
os.makedirs(OUT, exist_ok=True)

TARGET_FACES = 12000        # decimation target (from ~95k) -> smooth 3x WebGL
TEX_MAX = 1024              # cap texture size

OBJECTS = {
    "84": "Skimmer",
    "91": "Glass_Green",
    "95": "Scoop",
}


def load_textured(d):
    obj = f"{SRC}/{d}/model.obj"
    tex = f"{SRC}/{d}/texture.png"
    m = trimesh.load(obj, process=False, force="mesh")
    uv = getattr(m.visual, "uv", None)
    img = None
    if os.path.exists(tex):
        img = Image.open(tex).convert("RGB")
        if max(img.size) > TEX_MAX:
            s = TEX_MAX / max(img.size)
            img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)
    return m, uv, img


for d, name in OBJECTS.items():
    try:
        m, uv, img = load_textured(d)
        n0 = len(m.faces)
        # decimate (quadric) if the backend is available; keep UVs via slot-preserving fallback
        try:
            md = m.simplify_quadric_decimation(face_count=TARGET_FACES)
            # quadric decimation drops UVs -> re-bake vertex colors by sampling texture at
            # nearest original UV is overkill; instead only decimate when UVs survive.
            if img is not None and getattr(md.visual, "uv", None) is None and uv is not None:
                md = None  # cannot keep texture -> fall back to original
        except Exception:
            md = None
        if md is not None and len(md.faces) < n0:
            m = md
        # (re)attach texture
        cur_uv = getattr(m.visual, "uv", None)
        if img is not None and cur_uv is not None:
            m.visual = trimesh.visual.TextureVisuals(
                uv=cur_uv,
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=img, metallicFactor=0.0, roughnessFactor=0.75))
            tex_ok = True
        else:
            tex_ok = (getattr(m.visual, "material", None) is not None)
        m.apply_translation(-m.bounding_box.centroid)
        m.apply_scale(1.0 / max(m.extents))
        out = f"{OUT}/obj_{d}.glb"
        m.export(out)
        kb = os.path.getsize(out) / 1024
        print(f"[out] {out} ({name}) faces {n0}->{len(m.faces)} tex={tex_ok} {kb:.0f}KB")
    except Exception as e:
        print(f"[err] {d} {name}: {type(e).__name__}: {e}")

print("done.")
