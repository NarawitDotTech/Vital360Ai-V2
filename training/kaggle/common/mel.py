"""Log-mel front end that MUST match src/inference/mel.ts (MEL_DEFAULTS + logMel):
16 kHz, n_fft 1024, hop 320, periodic Hann (win 1024), centre=True with reflect padding, power=2,
64 mel bands 0..8000 Hz, Slaney scale + Slaney area norm, out = log(1e-10 + mel_power),
layout [frames][mels]; a 10 s clip (160000 samples) gives 1 + 160000//320 = 501 frames (= ort.ts default T)."""
import json

import librosa
import numpy as np
import torch

SR, N_FFT, HOP, N_MELS, EPS = 16000, 1024, 320, 64, 1e-10
FRAMES = 501
CLIP = (FRAMES - 1) * HOP  # 160000 samples = 10 s


class LogMel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT, periodic=True))
        fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=SR / 2, htk=False, norm="slaney")
        self.register_buffer("fb", torch.from_numpy(fb.astype(np.float32)))

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, L] float32 at 16 kHz -> [B, 1, frames, 64]."""
        spec = torch.stft(wav.float(), N_FFT, hop_length=HOP, win_length=N_FFT, window=self.window,
                          center=True, pad_mode="reflect", return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2
        mel = torch.matmul(self.fb, power)  # [B, 64, T]
        return torch.log(EPS + mel).transpose(1, 2).unsqueeze(1)


def write_fixture(path, n_samples: int = 16000) -> dict:
    """Deterministic test signal + expected log-mel, for a vitest parity test against mel.ts (tolerance ~5e-3)."""
    rng = np.random.default_rng(0)
    t = np.arange(n_samples) / SR
    x = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * (1500 + 800 * t) * t) + 0.01 * rng.standard_normal(n_samples)
    x = np.round(x, 6).astype(np.float32)
    m = LogMel()(torch.from_numpy(x)[None])[0, 0].numpy()
    fx = {"sampleRate": SR, "nFrames": int(m.shape[0]), "nMels": int(m.shape[1]),
          "samples": x.tolist(), "expected": np.round(m.reshape(-1), 5).tolist()}
    with open(path, "w") as f:
        json.dump(fx, f)
    return fx
