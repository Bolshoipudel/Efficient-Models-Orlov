import urllib.request
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/{}.bin"
DATA = Path(__file__).resolve().parent / "data"
HEADER_BYTES = 1024
MAGIC = 20240520
TRAIN_SHARDS = ["fineweb_train_000001", "fineweb_train_000002"]
VAL_SHARD = "fineweb_val_000000"
VAL_KEEP_TOKENS = 2**22


def download(name, n_bytes=None):
    path = DATA / f"{name}.bin"
    if path.exists():
        return path
    request = urllib.request.Request(URL.format(name))
    if n_bytes:
        request.add_header("Range", f"bytes=0-{n_bytes - 1}")
    print(f"скачиваю {name} ...", flush=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(request) as response, open(tmp, "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


def describe(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == MAGIC, f"{path.name}: неожиданный заголовок"
    tokens = np.fromfile(path, dtype=np.uint16, offset=HEADER_BYTES)
    print(f"{path.name}: {len(tokens):,} токенов, max id {tokens.max()}")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for shard in TRAIN_SHARDS:
        describe(download(shard))
    describe(download(VAL_SHARD, HEADER_BYTES + 2 * VAL_KEEP_TOKENS))
