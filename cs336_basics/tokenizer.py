from .pretokenization_example import train_bpe
from collections.abc import Iterable, Iterator
import regex as re
from itertools import pairwise
import json

class Tokenizer:
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        special_tokens = sorted(set(special_tokens or []), key=len, reverse=True)
        existing_tokens = set(self.vocab.values())
        next_id = max(self.vocab, default=-1) + 1

        for special_token in special_tokens:
            special_token = special_token.encode()
            if special_token not in existing_tokens:
                self.vocab[next_id] = special_token
                existing_tokens.add(special_token)
                next_id += 1

        self.inv_vocab = {value: key for key, value in self.vocab.items()}
        self.merge_priority = {merge: i for i, merge in enumerate(merges)}
        self.merges = merges
        self.special_pattern = (
            re.compile("|".join(re.escape(token) for token in special_tokens))
            if special_tokens else None
        )

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """Load JSON files with integer vocabulary keys and hex-encoded token bytes.

        The merges file is an ordered list of two-element lists of hex strings.
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_data = json.load(f)
        with open(merges_filepath, encoding="utf-8") as f:
            merges_data = json.load(f)

        vocab = {int(token_id): bytes.fromhex(token) for token_id, token in vocab_data.items()}
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in merges_data]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if not self.special_pattern:
            return self._encode_ordinary(text)
        ids = []
        position = 0

        for match in self.special_pattern.finditer(text):
            ids.extend(self._encode_ordinary(text[position:match.start()]))
            ids.append(self.inv_vocab[text[match.start():match.end()].encode()])
            position = match.end()
        ids.extend(self._encode_ordinary(text[position:]))
        return ids
    def _encode_ordinary(self, text: str) -> list[int]:
        it = re.finditer(self.PAT, text)
        ids = []
        # for re.split(self.str_split_pattern, text)
        for match in it:
            # We chose a n ** 2 algorithm because pre-tokens are generally small
            curr_text = [bytes([x]) for x in match.group().encode()]
            n = len(curr_text)
            if n == 0:
                continue
            while True:
                i, merge_idx = min(((self.merge_priority.get(pair, float('inf')), i) for i, pair in enumerate(pairwise(curr_text))), default=(float('inf'), None))
                if i == float('inf'):
                    ids.extend(self.inv_vocab[x] for x in curr_text)
                    break
                a, b = self.merges[i]
                curr_text = curr_text[:merge_idx] + [a + b] + curr_text[merge_idx + 2:]

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b''.join(self.vocab[id] for id in ids)
        return text.decode(errors='replace')
