"""SPRSound (HF mirror Treza12/sprsound-classification, CC BY 4.0 original) -> log-mel canvases. Record-level labels:
Normal -> normal, CAS (continuous adventitious: wheeze/rhonchi/stridor) -> wheeze, DAS (discontinuous: crackles) -> crackle.
'CAS & DAS' and 'Poor Quality' recordings are dropped. Usage: python audio_prep.py DATA_DIR CACHE_DIR"""
import struct, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.io import wavfile
from scipy.signal import resample_poly
from audio_mel import SR, log_mel, to_canvas


def read_wav(b: bytes):
    """The mirror's WAV headers have a wrong nBlockAlign (4 for mono 16-bit), so scipy refuses them: parse manually.
    Layout checked: PCM, 1 channel, 8000 Hz, 16 bit, 44-byte header (the SPRSound originals are 8 kHz)."""
    ch, sr, bits = struct.unpack("<H", b[22:24])[0], struct.unpack("<I", b[24:28])[0], struct.unpack("<H", b[34:36])[0]
    assert ch == 1 and bits == 16 and b[36:40] == b"data", (ch, sr, bits)
    return sr, np.frombuffer(b[44:], dtype="<i2").astype(np.float64) / 32768.0


MAP = {3: 0, 0: 1, 2: 2}  # parquet class id -> normal / wheeze / crackle


def main(data, cache):
    cache.mkdir(parents=True, exist_ok=True)
    X, rows = [], []
    for s in ("s0.parquet", "s1.parquet"):
        pf = pq.ParquetFile(data / s)
        for g in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(g, columns=["audio", "filename", "patient_num", "age", "gender", "disease", "label"]).to_pylist():
                if r["label"] not in MAP:
                    continue
                sr, x = read_wav(r["audio"]["bytes"])
                if sr != SR:
                    x = resample_poly(x, SR, sr)
                X.append(to_canvas(log_mel(x)))
                rows.append(dict(filename=r["filename"], group=r["patient_num"], age=r["age"], gender=r["gender"], disease=r["disease"], y=MAP[r["label"]], secs=len(x) / SR))
    pd.DataFrame(rows).to_csv(cache / "audio_index.csv", index=False)
    np.save(cache / "audio_mel.npy", np.stack(X))
    print(len(rows), pd.DataFrame(rows).y.value_counts().to_dict(), pd.DataFrame(rows).secs.describe().to_dict())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
