"""Python port of src/inference/mel.ts (logMel). MUST stay numerically equal to the TS (checked by src/inference/mel.test.ts via
src/inference/__fixtures__/mel_fixture.json): 16 kHz, n_fft 1024, hop 320, periodic Hann, centre reflect padding, power spectrum,
64 Slaney bands (Slaney scale + area norm, 0..8000 Hz), out = log(1e-10 + mel power), layout [frames][mels]."""
import numpy as np, torch

SR, N_FFT, HOP, N_MELS, EPS, FRAMES = 16000, 1024, 320, 64, 1e-10, 501
F_SP = 200 / 3
MIN_LOG_HZ = 1000.0
MIN_LOG_MEL = MIN_LOG_HZ / F_SP
LOG_STEP = np.log(6.4) / 27


def hz_to_mel(hz):
    return MIN_LOG_MEL + np.log(hz / MIN_LOG_HZ) / LOG_STEP if hz >= MIN_LOG_HZ else hz / F_SP


def mel_to_hz(m):
    return MIN_LOG_HZ * np.exp(LOG_STEP * (m - MIN_LOG_MEL)) if m >= MIN_LOG_MEL else m * F_SP


def filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS):
    lo, hi = hz_to_mel(0.0), hz_to_mel(sr / 2)
    pts = [mel_to_hz(lo + (hi - lo) * i / (n_mels + 1)) for i in range(n_mels + 2)]
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        up = (f - pts[m]) / (pts[m + 1] - pts[m]); down = (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1])
        fb[m] = np.maximum(0, np.minimum(up, down)) * 2 / (pts[m + 2] - pts[m])
    return fb


_FB = torch.from_numpy(filterbank())
_WIN = torch.hann_window(N_FFT, periodic=True, dtype=torch.float64)


def log_mel(x: np.ndarray) -> np.ndarray:
    """float samples at 16 kHz -> [frames, 64] float32 (frames = 1 + len//320)."""
    t = torch.from_numpy(np.asarray(x, dtype=np.float64))[None]
    spec = torch.stft(t, N_FFT, hop_length=HOP, win_length=N_FFT, window=_WIN, center=True, pad_mode="reflect", return_complex=True)
    power = spec.real ** 2 + spec.imag ** 2
    mel = _FB @ power[0]
    return torch.log(EPS + mel).T.numpy().astype(np.float32)


def to_canvas(m: np.ndarray, frames=FRAMES) -> np.ndarray:
    """Crop / pad with log(eps) exactly as src/inference/ort.ts does."""
    out = np.full((frames, N_MELS), np.log(EPS), dtype=np.float32)
    n = min(len(m), frames)
    out[:n] = m[:n]
    return out
