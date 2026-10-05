"""Train the Coralog Codas model.

Real data:  python train.py --data data      (folders per clan, or data/labels.csv)
No data:    python train.py                  (falls back to synthetic codas so it always runs)
"""
import argparse, csv, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, normalized_mutual_info_score, silhouette_score
from codanet import CodaNet, EXT, CODAS, read_audio, to_mel, synth


def load_real(root):
    """Folder layout data/<clan>/*.wav, or data/labels.csv with columns file,clan[,context][,recording]."""
    root, rows = Path(root), []
    if (root / "labels.csv").exists():
        with open(root / "labels.csv", newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append((root / r["file"].strip(), r["clan"].strip(),
                             (r.get("context") or "").strip(), (r.get("recording") or "").strip()))
    else:
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for w in sorted(d.rglob("*")):
                if w.suffix.lower() in EXT:
                    rows.append((w, d.name, "", ""))
    waves, keep = [], []
    for r in rows:
        try:
            waves.append(read_audio(r[0])); keep.append(r)
        except Exception as e:
            print("skipped", r[0], "-", e)
    if len(waves) < 8:
        raise SystemExit(f"Found only {len(waves)} readable audio files in '{root}'. Need at least 8.")
    clans = sorted({r[1] for r in keep})
    ctxs = sorted({r[2] for r in keep}) if all(r[2] for r in keep) else []
    if len(ctxs) < 2:
        ctxs = []
    yc = torch.tensor([clans.index(r[1]) for r in keep])
    yx = torch.tensor([ctxs.index(r[2]) for r in keep]) if ctxs else None
    groups = [r[3] or str(r[0]) for r in keep]
    if not any(r[3] for r in keep):
        print("Tip: add a 'recording' column to labels.csv so clips from one recording stay together in the split.")
    return waves, clans, yc, ctxs, yx, groups


def load_synth(rng, n=1200):
    ids = rng.integers(0, len(CODAS), n)
    clan_of = [0, 1, 1, 2]
    ctx = np.where(rng.random(n) < .75, ids, rng.integers(0, 4, n))
    waves = [synth(int(i), rng) for i in ids]
    return (waves, ["Clan A", "Clan B", "Clan C"], torch.tensor([clan_of[i] for i in ids]),
            ["socialising", "foraging", "travelling", "resting"], torch.tensor(ctx),
            [str(i // 8) for i in range(n)])


def split(groups, seed=0):
    """80/20 split by group (recording) so one recording never lands in both sets."""
    ug = sorted(set(groups)); random.Random(seed).shuffle(ug)
    test = set(ug[:max(1, int(.2 * len(ug)))]) if len(ug) > 1 else set()
    return ([i for i, g in enumerate(groups) if g not in test], [i for i, g in enumerate(groups) if g in test])


@torch.no_grad()
def run(model, X, ids, dev, bs=64):
    model.eval(); pc, px, es, ix = [], [], [], []
    for b in ids.split(bs):
        lc, lx, e, idx, _ = model(X[b].to(dev))
        pc.append(lc.argmax(1).cpu()); es.append(e.cpu()); ix.append(idx.cpu())
        if lx is not None:
            px.append(lx.argmax(1).cpu())
    return torch.cat(pc), (torch.cat(px) if px else None), torch.cat(es).numpy(), torch.cat(ix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-4)
    a = ap.parse_args()
    rng = np.random.default_rng(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if Path(a.data).is_dir():
        print(f"Loading audio from '{a.data}' ..."); waves, clans, yc, ctxs, yx, groups = load_real(a.data)
    else:
        print(f"No '{a.data}' folder found -> using SYNTHETIC codas (pipeline demo only)")
        waves, clans, yc, ctxs, yx, groups = load_synth(rng)
    print(f"{len(waves)} clips | clans: {clans} | contexts: {ctxs or 'none'} | device: {dev}")
    X = torch.stack([to_mel(w) for w in waves])
    tr, te = split(groups)
    if not te:
        te = tr; print("Warning: too few recordings for a test split; evaluating on training data.")
    tri, tei = torch.tensor(tr), torch.tensor(te)

    model = CodaNet(len(clans), len(ctxs)).to(dev)
    cb = model.codebook.num_embeddings
    cnt = torch.bincount(yc[tri], minlength=len(clans)).float().clamp(min=1)
    wts = (cnt.sum() / (len(clans) * cnt)).to(dev)               # balance rare clans
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-2)

    for ep in range(a.epochs):
        model.train(); tot = 0.0; used = torch.zeros(cb, device=dev)
        for b in tri[torch.randperm(len(tri))].split(a.bs):
            lc, lx, _, idx, vq = model(X[b].to(dev))
            loss = F.cross_entropy(lc, yc[b].to(dev), weight=wts) + vq
            if lx is not None:
                loss = loss + F.cross_entropy(lx, yx[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); used += torch.bincount(idx.flatten(), minlength=cb)
        dead = (used == 0).nonzero().flatten()                  # revive unused codewords (avoids codebook collapse)
        if len(dead) and ep < a.epochs // 2:
            z = model.last_z
            model.codebook.weight.data[dead] = z[torch.randint(len(z), (len(dead),), device=z.device)]
        pc, _, _, _ = run(model, X, tei, dev)
        print(f"epoch {ep + 1:2d}  loss {tot / len(tri):.3f}  test clan acc {(pc == yc[tei]).float().mean():.3f}")

    pc, px, e, idx = run(model, X, tei, dev)
    yt = yc[tei]
    print("\n=== Results (held-out test set) ===")
    print(f"clan accuracy {(pc == yt).float().mean():.3f} | macro-F1 {f1_score(yt, pc, average='macro', zero_division=0):.3f}")
    print("confusion matrix (rows = true clan, cols = predicted):\n", confusion_matrix(yt, pc, labels=range(len(clans))))
    if px is not None:
        base = torch.bincount(yx[tei]).max().item() / len(tei)
        print(f"context accuracy {(px == yx[tei]).float().mean():.3f} (majority-class baseline {base:.3f})")
    k = min(len(clans), len(e))
    if k >= 2:
        lab = KMeans(k, n_init=10, random_state=0).fit(e).labels_
        print(f"dialect clustering NMI {normalized_mutual_info_score(yt, lab):.3f}", end="")
        if 2 <= len(set(lab)) <= len(e) - 1:
            print(f" | silhouette {silhouette_score(e, lab):.3f}", end="")
        print()
    print(f"distinct tokens used {idx.unique().numel()} of {cb}")

    if len(e) >= 2:
        p = PCA(2).fit_transform(e)
        plt.figure(figsize=(6, 5))
        for c, n in enumerate(clans):
            m = (yt == c).numpy()
            plt.scatter(p[m, 0], p[m, 1], s=18, label=n)
        plt.legend(); plt.title("Dialect map (PCA of Transformer embeddings)"); plt.tight_layout()
        plt.savefig("dialect_map.png", dpi=150); plt.close()
    torch.save({"state": model.state_dict(), "clans": clans, "contexts": ctxs,
                "n_clan": len(clans), "n_ctx": len(ctxs)}, "codanet.pt")
    np.save("embeddings.npy", e)
    print("Saved: codanet.pt, embeddings.npy, dialect_map.png")


if __name__ == "__main__":
    main()
