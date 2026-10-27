"""Artifact conversion, tokenizer measurements, and resumable dataset encoding.

Run from the repository root with `uv run python scripts/tokenizer_experiments.py --help`.
Tokenization itself is delegated to the student's Tokenizer implementation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cs336_basics.tokenizer import Tokenizer

ARTIFACTS = ROOT / "artifacts"
DELIMITER = b"<|endoftext|>"
DATASETS = {
    "tinystories": ("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"),
    "owt": ("owt_train.txt", "owt_valid.txt"),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def documents(path, start=0, block_size=1024 * 1024):
    """Yield (end byte offset, document bytes), retaining document delimiters.

    Splitting only at special-token boundaries preserves ordinary pre-token context.
    Memory is bounded by the current document plus one read block.
    """
    offset = start
    pending = b""
    with Path(path).open("rb") as handle:
        handle.seek(start)
        while block := handle.read(block_size):
            parts = (pending + block).split(DELIMITER)
            pending = parts.pop()
            for part in parts:
                document = part + DELIMITER
                offset += len(document)
                yield offset, document
        if pending:
            yield offset + len(pending), pending


def tokenizer_path(dataset):
    return ARTIFACTS / "tokenizers" / dataset


def load_tokenizer(dataset):
    folder = tokenizer_path(dataset)
    return Tokenizer.from_files(folder / "vocab.json", folder / "merges.json", [DELIMITER.decode()])


def convert(dataset, source):
    # Only load locally produced, trusted training artifacts.
    with (source / "tokenizer.pkl").open("rb") as handle:
        vocab, merges = pickle.load(handle)
    folder = tokenizer_path(dataset)
    write_json(folder / "vocab.json", {key: value.hex() for key, value in vocab.items()})
    write_json(folder / "merges.json", [[a.hex(), b.hex()] for a, b in merges])
    loaded = load_tokenizer(dataset)
    assert loaded.vocab == vocab and loaded.merges == merges
    metrics = json.loads((source / "metrics.json").read_text())
    longest = max(vocab.values(), key=len)
    metrics.update(longest_token_bytes=len(longest), longest_token_display=longest.decode(errors="replace"))
    write_json(folder / "training_metrics.json", metrics)
    shutil.copyfile(source / "time.txt", folder / "training_time.txt")
    print(f"Converted and verified {dataset}: {len(vocab)} tokens, {len(merges)} merges", flush=True)


def sample_documents(path, count=10, pool_size=1000, seed=336):
    """Seeded sample of ten nonempty documents from the first 1,000 documents."""
    pool = []
    previous_end = 0
    for end, document in documents(path):
        start = previous_end
        previous_end = end
        if document.removesuffix(DELIMITER).strip():
            pool.append((start, end, document))
        if len(pool) >= pool_size:
            break
    return random.Random(seed).sample(pool, min(count, len(pool)))


def measure():
    samples = {}
    for dataset, filenames in DATASETS.items():
        path = ROOT / "data" / filenames[0]
        samples[dataset] = sample_documents(path)
        write_json(ARTIFACTS / "experiments" / f"{dataset}_samples.json", {
            "source": str(path), "seed": 336, "sampling": "10 from first 1000 nonempty documents",
            "documents": [{"start": start, "end": end, "text": data.decode("utf-8")}
                          for start, end, data in samples[dataset]],
        })

    results = []
    for model, corpus in [("tinystories", "tinystories"), ("owt", "owt"), ("tinystories", "owt")]:
        tokenizer = load_tokenizer(model)
        texts = [data.decode("utf-8") for _, _, data in samples[corpus]]
        byte_count = sum(len(data) for _, _, data in samples[corpus])
        tokenizer.encode(texts[0])  # warm-up excluded from timing
        times = []
        for _ in range(3):
            start = time.perf_counter()
            encoded = [tokenizer.encode(text) for text in texts]
            times.append(time.perf_counter() - start)
        for text, ids in zip(texts, encoded):
            assert tokenizer.decode(ids) == text
        token_count = sum(map(len, encoded))
        seconds = sorted(times)[1]
        row = {
            "tokenizer": model, "corpus": corpus, "documents": len(texts),
            "bytes": byte_count, "tokens": token_count, "bytes_per_token": byte_count / token_count,
            "encoding_seconds": times, "median_seconds": seconds,
            "bytes_per_second": byte_count / seconds,
            "estimated_pile_hours_825_decimal_GB": 825e9 / (byte_count / seconds) / 3600,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    write_json(ARTIFACTS / "experiments" / "measurements.json", results)


def signature(dataset, source):
    folder = tokenizer_path(dataset)
    digest = hashlib.sha256()
    for path in [folder / "vocab.json", folder / "merges.json", ROOT / "cs336_basics/tokenizer.py"]:
        digest.update(path.read_bytes())
    stat = source.stat()
    return {"source": str(source), "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
            "tokenizer_sha256": digest.hexdigest(), "dtype": "<u2"}


def encode_dataset(dataset, split):
    source = ROOT / "data" / DATASETS[dataset][split == "valid"]
    folder = ARTIFACTS / "tokens"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{dataset}_{split}"
    raw_path = stem.with_suffix(".partial.bin")
    checkpoint = stem.with_suffix(".progress.json")
    final_path = stem.with_suffix(".npy")
    metadata = stem.with_suffix(".json")
    identity = signature(dataset, source)
    if final_path.exists() and metadata.exists():
        assert json.loads(metadata.read_text())["identity"] == identity, "Existing output uses different inputs"
        print(f"Already complete: {final_path}", flush=True)
        return
    state = {"identity": identity, "offset": 0, "tokens": 0, "documents": 0, "encoding_seconds": 0.0}
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        assert state["identity"] == identity, "Resume inputs changed; use a fresh output directory"
    tokenizer = load_tokenizer(dataset)
    assert min(tokenizer.vocab) >= 0 and max(tokenizer.vocab) <= np.iinfo(np.uint16).max
    started = last_save = time.perf_counter()
    prior_seconds = state["encoding_seconds"]
    with raw_path.open("r+b" if raw_path.exists() else "w+b") as output:
        if os.fstat(output.fileno()).st_size < state["tokens"] * 2:
            raise ValueError("Partial output is shorter than its checkpoint")
        output.truncate(state["tokens"] * 2)
        output.seek(0, os.SEEK_END)

        def save_progress():
            output.flush()
            os.fsync(output.fileno())
            state["encoding_seconds"] = prior_seconds + time.perf_counter() - started
            write_json(checkpoint, state)
            fraction = state["offset"] / identity["source_size"]
            print(f"{dataset}/{split}: {fraction:.1%}, {state['tokens']:,} tokens, "
                  f"{state['encoding_seconds']:.1f}s", flush=True)

        for end, data in documents(source, start=state["offset"]):
            ids = tokenizer.encode(data.decode("utf-8"))
            np.asarray(ids, dtype="<u2").tofile(output)
            state.update(offset=end, tokens=state["tokens"] + len(ids), documents=state["documents"] + 1)
            if time.perf_counter() - last_save >= 30:
                save_progress()
                last_save = time.perf_counter()
        save_progress()

    temporary = stem.with_suffix(".partial.npy")
    target = np.lib.format.open_memmap(temporary, mode="w+", dtype="<u2", shape=(state["tokens"],))
    with raw_path.open("rb") as handle:
        position = 0
        while len(block := np.fromfile(handle, dtype="<u2", count=4 * 1024 * 1024)):
            target[position:position + len(block)] = block
            position += len(block)
    target.flush()
    del target
    assert position == state["tokens"]
    temporary.replace(final_path)
    write_json(metadata, {**state, "output": str(final_path), "complete": True})
    raw_path.unlink()
    print(f"Complete: {final_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    converter = sub.add_parser("convert", help="Convert a trusted local benchmark pickle to hex JSON")
    converter.add_argument("dataset", choices=DATASETS)
    converter.add_argument("source", type=Path)
    sub.add_parser("measure", help="Run reproducible compression and throughput experiments")
    encoder = sub.add_parser("encode", help="Resume encoding a complete dataset split into a uint16 .npy")
    encoder.add_argument("dataset", choices=DATASETS)
    encoder.add_argument("split", choices=["train", "valid"])
    args = parser.parse_args()
    if args.command == "convert":
        convert(args.dataset, args.source)
    elif args.command == "measure":
        measure()
    else:
        encode_dataset(args.dataset, args.split)


if __name__ == "__main__":
    main()
