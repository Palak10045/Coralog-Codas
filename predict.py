"""Predict the clan / context of one audio file:  python predict.py my_recording.wav"""
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from codanet import CodaNet, read_audio, to_mel

ap = argparse.ArgumentParser()
ap.add_argument("wav"); ap.add_argument("--model", default="codanet.pt")
a = ap.parse_args()

ck = torch.load(a.model, map_location="cpu")
model = CodaNet(ck["n_clan"], ck["n_ctx"])
model.load_state_dict(ck["state"]); model.eval()

mel = to_mel(read_audio(a.wav))
with torch.no_grad():
    lc, lx, _, idx, _ = model(mel.unsqueeze(0))

pc = torch.softmax(lc[0], 0)
print(f"\nFile: {a.wav}\n\nClan / dialect prediction:")
for i in pc.argsort(descending=True).tolist():
    print(f"  {ck['clans'][i]:<20s} {pc[i] * 100:5.1f}%")
px = None
if lx is not None:
    px = torch.softmax(lx[0], 0)
    print("\nBehavioural context prediction (exploratory):")
    for i in px.argsort(descending=True).tolist():
        print(f"  {ck['contexts'][i]:<20s} {px[i] * 100:5.1f}%")
print("\nFirst 30 acoustic tokens:", idx[0][:30].tolist())
print("\nNote: the model only knows the classes it was trained on. Treat results as suggestions.")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2, 1]})
ax[0].imshow(mel[0].numpy(), origin="lower", aspect="auto", cmap="viridis")
ax[0].set_title("Log-mel spectrogram"); ax[0].set_xlabel("time frames"); ax[0].set_ylabel("mel band")
ax[1].barh(ck["clans"], (pc * 100).tolist()); ax[1].set_xlim(0, 100)
ax[1].set_title("Clan probability (%)")
plt.tight_layout(); plt.savefig("prediction.png", dpi=150)
print("Saved: prediction.png")
