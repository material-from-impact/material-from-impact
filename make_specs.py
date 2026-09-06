"""
Generate clean, theme-agnostic log-mel spectrogram panels for the project page.

For each demo we render a single PNG holding TWO mel panels side by side
(input | re-render), with NO axes / NO title / NO colorbar and a TRANSPARENT
background, so the same image reads on both the light and dark page themes.
The "Input / Re-render" labels are added in HTML, not baked in.

Onset-windowed (~1.4 s from the first strike) and fixed dB scale for
cross-panel comparability.

Run (either env with librosa; e.g. niat):
    python gitpage/make_specs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

SR = 44100
NFFT, HOP, NMELS = 2048, 512, 128
DB_LO, DB_HI = -80.0, 0.0

AUD = "gitpage/assets/audio"
OUT = "gitpage/assets/spec"
os.makedirs(OUT, exist_ok=True)

# (out_name, input_wav, rerender_wav, window_seconds)
PAIRS = [
    ("syn_Ceramic", f"{AUD}/syn_Ceramic_input.wav", f"{AUD}/syn_Ceramic_pred.wav", 0.6),
    ("syn_Glass",   f"{AUD}/syn_Glass_input.wav",   f"{AUD}/syn_Glass_pred.wav",   0.6),
    ("syn_Plastic", f"{AUD}/syn_Plastic_input.wav", f"{AUD}/syn_Plastic_pred.wav", 0.6),
    ("of_84_Skimmer",    f"{AUD}/of_84_Skimmer_input.wav",    f"{AUD}/of_84_Skimmer_pred.wav",    1.2),
    ("of_91_Glass_Green",f"{AUD}/of_91_Glass_Green_input.wav",f"{AUD}/of_91_Glass_Green_pred.wav",1.2),
    ("of_95_Scoop",      f"{AUD}/of_95_Scoop_input.wav",      f"{AUD}/of_95_Scoop_pred.wav",      1.2),
]


def load_win(path, win):
    y, _ = librosa.load(path, sr=SR, mono=True)
    n = int(win * SR)
    if y.size == 0:
        return np.zeros(n, dtype=np.float32)
    e = np.abs(y); m = e.max()
    idx = np.where(e > 0.02 * m)[0] if m > 0 else np.array([])
    s0 = max(0, (idx[0] - int(0.02 * SR)) if idx.size else 0)
    seg = y[s0:s0 + n]
    if seg.size < n:
        seg = np.pad(seg, (0, n - seg.size))
    return seg


def logmel(y):
    S = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=SR, n_fft=NFFT,
                                       hop_length=HOP, n_mels=NMELS)
    return librosa.power_to_db(S + 1e-10, ref=np.max)


for name, fin, frr, win in PAIRS:
    if not (os.path.exists(fin) and os.path.exists(frr)):
        print(f"[skip] {name}: missing wav")
        continue
    yi, yr = load_win(fin, win), load_win(frr, win)
    Mi, Mr = logmel(yi), logmel(yr)
    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.15))
    for a, M in [(ax[0], Mi), (ax[1], Mr)]:
        a.imshow(M, origin="lower", aspect="auto", cmap="magma",
                 vmin=DB_LO, vmax=DB_HI, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.03)
    fig.savefig(f"{OUT}/{name}.png", dpi=170, transparent=True,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[out] {OUT}/{name}.png")

print("done.")
