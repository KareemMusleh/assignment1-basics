import json

import numpy as np

from scripts import tokenizer_experiments as experiments


def test_document_stream_preserves_split_delimiters_and_utf8(tmp_path):
    payload = "café<|endoftext|><|endoftext|>🙃 trailing".encode()
    path = tmp_path / "input.txt"
    path.write_bytes(payload)
    rows = list(experiments.documents(path, block_size=3))
    assert b"".join(data for _, data in rows) == payload
    assert [data for _, data in rows] == ["café<|endoftext|>".encode(), experiments.DELIMITER, "🙃 trailing".encode()]
    assert rows[-1][0] == len(payload)
    assert list(experiments.documents(path, start=rows[0][0], block_size=2)) == rows[1:]


def test_encoding_resumes_only_committed_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "ROOT", tmp_path)
    monkeypatch.setattr(experiments, "ARTIFACTS", tmp_path / "artifacts")
    (tmp_path / "cs336_basics").mkdir()
    (tmp_path / "cs336_basics/tokenizer.py").write_text("test identity")
    (tmp_path / "data").mkdir()
    source = tmp_path / "data" / experiments.DATASETS["tinystories"][0]
    text = "café<|endoftext|>🙃 tail"
    source.write_bytes(text.encode())
    folder = experiments.tokenizer_path("tinystories")
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = experiments.DELIMITER
    experiments.write_json(folder / "vocab.json", {i: value.hex() for i, value in vocab.items()})
    experiments.write_json(folder / "merges.json", [])
    tokenizer = experiments.load_tokenizer("tinystories")
    end, document = next(experiments.documents(source))
    ids = tokenizer.encode(document.decode())
    stem = experiments.ARTIFACTS / "tokens/tinystories_train"
    experiments.write_json(stem.with_suffix(".progress.json"), {
        "identity": experiments.signature("tinystories", source),
        "offset": end, "tokens": len(ids), "documents": 1, "encoding_seconds": 0,
    })
    np.asarray(ids + [42, 43], dtype="<u2").tofile(stem.with_suffix(".partial.bin"))
    experiments.encode_dataset("tinystories", "train")
    actual = np.load(stem.with_suffix(".npy"), mmap_mode="r")
    assert actual.dtype == np.dtype("<u2")
    assert actual.tolist() == tokenizer.encode(text)
    assert tokenizer.decode(actual.tolist()) == text
    assert json.loads(stem.with_suffix(".json").read_text())["complete"]
    experiments.encode_dataset("tinystories", "train")  # completed output is safely reused
