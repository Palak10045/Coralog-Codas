"""Shared model + audio helpers for Coralog Codas (used by train.py and predict.py)."""
import numpy as np, torch, torch.nn as nn, torchaudio
import soundfile as sf

SR, DUR, N_MELS = 16000, 2.0, 64          # sample rate, clip length (s), mel bands
EXT = {".wav", ".flac", ".ogg"}
MEL = torchaudio.transforms.MelSpectrogram(SR, n_fft=512, hop_length=160, n_mels=N_MELS)

# Synthetic coda rhythms (click times in seconds). Used only for demo/fallback data.
CODAS = [[0, .55, 1.25, 1.45, 1.65], [0, .22, .44, .66, .88], [0, .25, .5, .75], [0, .12, .26, .42, .6, .8, 1.04]]


def synth(i, rng):
    """One synthetic coda recording (float32 array) of rhythm type i."""
    x = rng.normal(0, .02, int(SR * DUR)).astype("float32")
    for t in CODAS[i]:
        n = max(int((t + rng.normal(0, .01)) * SR), 0)
        click = (rng.normal(0, 1, 400) * np.hanning(400) * 3).astype("float32")
        seg = x[n:n + 400]
        seg += click[:len(seg)]
    return x


def fit_length(x):
    n = int(SR * DUR)
    x = x[:n]
    return np.pad(x, (0, n - len(x))).astype("float32")


def read_audio(path):
    """Load wav/flac/ogg -> mono float32 at SR, padded or cropped to DUR seconds."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x.mean(1))
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return fit_length(x)


def to_mel(x):
    """Waveform (numpy) -> standardised log-mel spectrogram, shape (1, N_MELS, frames)."""
    m = torch.log(MEL(torch.from_numpy(np.ascontiguousarray(x))) + 1e-6)
    return ((m - m.mean()) / (m.std() + 1e-6)).unsqueeze(0)


class CodaNet(nn.Module):
    """CNN patch encoder -> vector-quantised tokens -> Transformer -> clan / context heads."""

    def __init__(s, n_clan, n_ctx=0, d=128, codebook=32):
        super().__init__()
        s.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, d, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((1, None)))
        s.codebook = nn.Embedding(codebook, d)                    # the acoustic "vocabulary"
        s.pos = nn.Parameter(torch.zeros(1, 512, d))
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        s.clan = nn.Linear(d, n_clan)
        s.ctx = nn.Linear(d, n_ctx) if n_ctx > 1 else None
        s.last_z = None

    def forward(s, m):
        z = s.cnn(m).squeeze(2).transpose(1, 2)                   # (B, T, d)
        assert z.size(1) <= 512, "clip too long for positional table"
        s.last_z = z.detach().reshape(-1, z.size(-1))
        idx = torch.cdist(z, s.codebook.weight).argmin(-1)        # tokenise: nearest codeword
        q = s.codebook(idx)
        vq = nn.functional.mse_loss(q, z.detach()) + .25 * nn.functional.mse_loss(z, q.detach())
        h = s.enc(z + (q - z).detach() + s.pos[:, :z.size(1)])    # straight-through estimator
        e = h.mean(1)
        return s.clan(e), (s.ctx(e) if s.ctx is not None else None), e, idx, vq
