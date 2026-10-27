import json

from cs336_basics.tokenizer import Tokenizer


def test_from_files_preserves_bytes_and_merge_order(tmp_path):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256: b"ab", 257: b"abc"})
    merges = [(b"a", b"b"), (b"ab", b"c")]
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.json"
    vocab_path.write_text(json.dumps({i: value.hex() for i, value in vocab.items()}))
    merges_path.write_text(json.dumps([[a.hex(), b.hex()] for a, b in merges]))

    tokenizer = Tokenizer.from_files(vocab_path, merges_path)
    assert tokenizer.vocab == vocab
    assert tokenizer.merges == merges
    assert tokenizer.encode("abc") == [257]
    assert tokenizer.decode(tokenizer.encode("abc café 🙃")) == "abc café 🙃"

    with_special = Tokenizer.from_files(vocab_path, merges_path, ["<special>"])
    assert with_special.encode("abc<special>") == [257, 258]
    assert with_special.decode([257, 258]) == "abc<special>"
