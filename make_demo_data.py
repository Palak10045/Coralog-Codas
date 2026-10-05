"""Create a small SYNTHETIC dataset in the same format real data should use.
Run once:  python make_demo_data.py   ->  creates data/ and sample_coda.wav
These are computer-made click patterns, not real whales. Use them to test the pipeline."""
import csv
from pathlib import Path
import numpy as np, soundfile as sf
from codanet import synth, SR

rng = np.random.default_rng(1)
root = Path("data")
clan = {0: "clan_A", 1: "clan_B", 2: "clan_B", 3: "clan_C"}
ctx = ["socialising", "foraging", "travelling", "resting"]


def save(path, x):
    sf.write(str(path), (x / np.abs(x).max() * .9).astype("float32"), SR)


rows = []
for n in range(800):
    i = int(rng.integers(0, 4))
    c = i if rng.random() < .75 else int(rng.integers(0, 4))
    rel = f"{clan[i]}/coda_{n:03d}.wav"
    (root / clan[i]).mkdir(parents=True, exist_ok=True)
    save(root / rel, synth(i, rng))
    rows.append([rel, clan[i], ctx[c], f"rec_{n // 6:02d}"])
with open(root / "labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["file", "clan", "context", "recording"]); w.writerows(rows)
save(Path("sample_coda.wav"), synth(1, rng))
print("Created data/ (800 clips + labels.csv) and sample_coda.wav")
