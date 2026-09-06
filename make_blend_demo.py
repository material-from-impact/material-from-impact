"""
Render a clean material-interpolation set for the project page's blend slider.

Same object, same strike (t_start / hit_index / gain fixed) across all weights, so
ONLY the material changes. Ceramic <-> Plastic at w = {0,.25,.5,.75,1} weight-on-Ceramic.
Each weight is a full FEM re-solve (E, rho change the modal frequencies).

Outputs:
  gitpage/assets/blend/blend_c{000,025,050,075,100}.wav
  gitpage/assets/blend/params.json   (per-weight interpolated physical params)

Env: niat (FEM deps), from repo root:
  python gitpage/make_blend_demo.py
"""
import os
import json
import numpy as np
import scipy.io.wavfile as wav

from dataset.constants import AUDIO_SAMPLE_RATE
from dataset.generate_blend_dataset import blended_params, fem_features, render_single

VOXEL = "training_dataset/1/voxel.npz"
OUT = "gitpage/assets/blend"
M1, M2 = "Ceramic", "Plastic"          # w = weight on M1 (Ceramic)
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
HIT_INDEX = 0
T_START = 0.04
GAIN = 1.0

os.makedirs(OUT, exist_ok=True)
voxel = np.load(VOXEL)["voxel"]

entries = []
for w in WEIGHTS:
    rho, E, nu, alpha, beta = blended_params(M1, M2, w)
    freqs, feats_in, _, _ = fem_features(voxel, (rho, E, nu, alpha, beta))
    y = render_single(freqs, feats_in, alpha, beta, HIT_INDEX, T_START, GAIN)
    pct = int(round(w * 100))
    name = f"blend_c{pct:03d}.wav"
    wav.write(f"{OUT}/{name}", AUDIO_SAMPLE_RATE, (y * 32767).astype(np.int16))
    entries.append({"pct_m1": pct, "wav": name,
                    "rho": float(rho), "E": float(E), "nu": float(nu),
                    "alpha": float(alpha), "beta": float(beta)})
    print(f"[out] {OUT}/{name}  {M1} {pct}% : rho={rho:.0f} E={E:.2e} "
          f"nu={nu:.3f} a={alpha:.2f} b={beta:.2e}", flush=True)

meta = {"m1": M1, "m2": M2, "object": "obj-1",
        "hit_index": HIT_INDEX, "t_start": T_START, "entries": entries}
json.dump(meta, open(f"{OUT}/params.json", "w"), indent=2)
print(f"[out] {OUT}/params.json")
