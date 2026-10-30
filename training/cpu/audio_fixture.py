"""Write src/inference/__fixtures__/mel_fixture.json: a synthetic signal and a real SPRSound clip (1 s) with the Python log-mel,
for the vitest parity test of src/inference/mel.ts. Usage: python audio_fixture.py DATA_DIR REPO_DIR"""
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from scipy.signal import resample_poly
from audio_mel import SR, log_mel
from audio_prep import read_wav


def case(name, x):
    x = np.round(x, 5).astype(np.float32)
    m = log_mel(x)
    return dict(name=name, sampleRate=SR, nFrames=int(m.shape[0]), nMels=int(m.shape[1]), samples=x.tolist(), expected=np.round(m.reshape(-1), 4).tolist())


def main(data, repo):
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    synth = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * (1500 + 800 * t) * t) + 0.01 * rng.standard_normal(SR)
    pf = pq.ParquetFile(data / "s0.parquet")
    r = pf.read_row_group(0, columns=["audio", "filename"]).to_pylist()[3]
    sr, x = read_wav(r["audio"]["bytes"])
    real = resample_poly(x, SR, sr)[SR : 2 * SR]
    out = repo / "src/inference/__fixtures__"; out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(cases=[case("synthetic", synth), case("sprsound-clip", real)]), open(out / "mel_fixture.json", "w"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
