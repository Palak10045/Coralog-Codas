"""Coralog Codas web app. Run:  python app.py   then open http://localhost:8000
Needs codanet.pt (run train.py first). Uses only the packages you already installed."""
import io, json, os, sys, webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn.functional as F, torchaudio
from sklearn.decomposition import PCA
from codanet import CodaNet, SR, DUR, fit_length, to_mel, synth

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
if not Path("codanet.pt").exists():
    sys.exit("codanet.pt not found. Run:  python train.py --data data   first.")
ck = torch.load("codanet.pt", map_location="cpu")
model = CodaNet(ck["n_clan"], ck["n_ctx"])
model.load_state_dict(ck["state"]); model.eval()
CLANS, CTXS = ck["clans"], ck["contexts"]


def audio_from_bytes(b):
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x.mean(1))
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return fit_length(x)


@torch.no_grad()
def analyse(x):
    mel = to_mel(x).unsqueeze(0)
    lc, lx, _, idx, _ = model(mel)
    z = model.cnn(mel).squeeze(2).transpose(1, 2)                 # real self-attention of the last layer
    h = model.codebook(idx) + model.pos[:, :z.size(1)]
    layers = model.enc.layers
    for l in layers[:-1]:
        h = l(h)
    a = layers[-1].norm1(h)
    att = layers[-1].self_attn(a, a, a, need_weights=True)[1][0]
    att = F.adaptive_avg_pool2d(att[None, None], (16, 16))[0, 0]
    m = mel[0, 0]
    lo, hi = torch.quantile(m, .02), torch.quantile(m, .995)
    m = ((m - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    out = {"probs": torch.softmax(lc[0], 0).tolist(), "tokens": idx[0].tolist(),
           "mel": [[round(v, 2) for v in r] for r in m.tolist()],
           "att": (att / att.max()).tolist(), "point": None}
    if lx is not None:
        out["cprobs"] = torch.softmax(lx[0], 0).tolist()
    emb = model(mel)[2].numpy()
    if PCA_MODEL is not None:
        out["point"] = PCA_MODEL.transform(emb)[0].tolist()
    return out


def build_map():
    if not Path("data").is_dir():
        return None, []
    try:
        from train import load_real
        waves, clans, yc, *_ = load_real("data")
        X = torch.stack([to_mel(w) for w in waves])
        with torch.no_grad():
            E = torch.cat([model(b)[2] for b in X.split(64)]).numpy()
        pca = PCA(2).fit(E); P = pca.transform(E)
        keep = np.random.default_rng(0).permutation(len(P))[:400]
        return pca, [[float(P[i, 0]), float(P[i, 1]), int(yc[i])] for i in keep]
    except Exception as e:
        print("Dialect map unavailable:", e)
        return None, []


print("Preparing dialect map from data/ ...")
PCA_MODEL, MAP = build_map()
META = {"clans": CLANS, "contexts": CTXS, "map": MAP}


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, (HERE / "app_ui.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/meta":
            self.send(200, json.dumps(META).encode())
        else:
            self.send(404, b"{}")

    def do_POST(self):
        try:
            if self.path.startswith("/sample/"):
                k, rng = self.path.rsplit("/", 1)[1], np.random.default_rng()
                x = fit_length(rng.normal(0, .2, int(SR * DUR)).astype("float32")) if k == "noise" else synth(int(k), rng)
            else:
                x = audio_from_bytes(self.rfile.read(int(self.headers["Content-Length"])))
            self.send(200, json.dumps(analyse(x)).encode())
        except Exception as e:
            self.send(400, json.dumps({"error": f"Could not analyse this audio: {e}"}).encode())

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    url = "http://localhost:8000"
    print(f"Ready. Open {url}  (press Ctrl+C to stop)")
    webbrowser.open(url)
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
