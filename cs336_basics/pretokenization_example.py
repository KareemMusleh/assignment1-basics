import os
from typing import BinaryIO
import regex as re
from itertools import pairwise
import json
from collections import defaultdict
import heapq

class MaxObj:
    def __init__(self, value):
        self.value = value
    def __lt__(self, other):
        return self.value > other.value
    def __eq__(self, other):
        return self.value == other.value

def find_chunk_boundaries(


    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: re.Pattern[bytes],
    max_special_token_len: int,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, re.Pattern), "Must represent special token as a bytestring"
    assert isinstance(split_special_token.pattern, bytes), "Must be a bytes regex"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size + max_special_token_len - 1)  # Read a mini chunk
            file.seek(initial_position + mini_chunk_size)

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            match = re.search(split_special_token, mini_chunk)
            # found_at = mini_chunk.find(split_special_token)
            if match is not None:
                found_at = match.start()
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size > len(special_tokens) + 256
    assert special_tokens # HACK

    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({i: token.encode() for i, token in enumerate(special_tokens, len(vocab))})
    merges: list[tuple[bytes, bytes]] = []
    num_processes = 4
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    str_split_pattern = re.compile('(?:' + '|'.join(map(lambda x: re.escape(x), special_tokens)) + ')')

    byte_special_tokens = list(map(lambda x: x.encode(), special_tokens))
    max_special_token_len = max(len(byte_special_token) for byte_special_token in byte_special_tokens)
    byte_split_pattern = re.compile(b'(?:' + b'|'.join(map(lambda x: re.escape(x), byte_special_tokens)) + b')')

    with open(input_path, "rb") as f:

        boundaries = find_chunk_boundaries(f, num_processes, byte_split_pattern, max_special_token_len)
        # assert len(boundaries) - 1 == num_processes

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        pretoken_counter = {}
        for start, end in pairwise(boundaries):
            # print(start, end)
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # print("READ THE CHUNK")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            # merges, vocab
            sub_chunks = re.split(str_split_pattern, chunk)
            for sub_chunk in sub_chunks:
                # it = re.finditer(PAT, sub_chunk)
                it = re.findall(PAT, sub_chunk)
                for match in it:
                    # pretoken = match.group(0)
                    pretoken = match
                    pretoken_counter[pretoken] = pretoken_counter.get(pretoken, 0) + 1

    pretoken_counter = {pretoken.encode(): pretoken_counter[pretoken] for pretoken in pretoken_counter}
    pretoken_tokens: dict[bytes, list[int]] = {pretoken: [x for x in pretoken] for pretoken in pretoken_counter}
    # print("FINISHED FILE PROCESSING")

    token_pair_counter = defaultdict(int)
    token_pair_pretokens = {}
    token_pair_heap = []

    def add_pairs_to_heap(pairs):
        for pair, count in pairs:
            heapq.heappush(token_pair_heap, (MaxObj(count), (MaxObj(vocab[pair[0]]), MaxObj(vocab[pair[1]])), pair))

    for pretoken, tokens in pretoken_tokens.items():
        pretoken_count = pretoken_counter[pretoken]
        for pair in pairwise(tokens):
            token_pair_counter[pair] += pretoken_count

        # We save the indices of the pairs to speed up iteration
        for pair in pairwise(tokens):
            token_pair_pretokens.setdefault(pair, set()).add(pretoken)


    add_pairs_to_heap(token_pair_counter.items())

    for new_token_id in range(len(vocab), vocab_size):
        found = False
        while token_pair_heap and not found:
            count, _, pair = heapq.heappop(token_pair_heap)
            count = count.value
            found = token_pair_counter[pair] == count
        if not found:
            break
        assert count >= 1
        byte_pair = (vocab[pair[0]], vocab[pair[1]])

        vocab[new_token_id] = vocab[pair[0]] + vocab[pair[1]]
        merges.append(byte_pair)

        # We have to calculate the new counts
        # For a pretoken "abcd" if we merge 'b' and 'c' we have to subtract 'ab' count and 'cd' count
        changes = {}
        for pretoken in token_pair_pretokens[pair]:
            tokens = pretoken_tokens[pretoken]
            prev_idx = -2
            for start, (first_token, second_token) in enumerate(pairwise(tokens)):
                if first_token != pair[0] or second_token != pair[1] or start - prev_idx == 1:
                    continue
                prev_idx = start
                changes.setdefault(pretoken, []).append(start)
                token_pair_counter[pair] -= pretoken_counter[pretoken]
        changed_pairs = set()
        for pretoken, indeces in changes.items():
            tokens = pretoken_tokens[pretoken]
            count = 0
            for start in sorted(indeces):
                start -= count
                count += 1
                tokens = tokens[:start] + tokens[start + 1:]
                tokens[start] = new_token_id
                if start - 1 >= 0:
                    prev_token = tokens[start - 1]
                    token_pair_pretokens.setdefault((prev_token, new_token_id), set()).add(pretoken)

                    token_pair_counter[prev_token, new_token_id] += pretoken_counter[pretoken]
                    changed_pairs.add((prev_token, new_token_id))
                    token_pair_counter[prev_token, pair[0]] -= pretoken_counter[pretoken]
                    changed_pairs.add((prev_token, pair[0]))

                if start + 1 < len(tokens):
                    next_token = tokens[start + 1]
                    token_pair_pretokens.setdefault((new_token_id, next_token), set()).add(pretoken)

                    token_pair_counter[new_token_id, next_token] += pretoken_counter[pretoken]
                    changed_pairs.add((new_token_id, next_token))
                    token_pair_counter[pair[1], tokens[start + 1]] -= pretoken_counter[pretoken]
                    changed_pairs.add((pair[1], tokens[start + 1]))

            pretoken_tokens[pretoken] = tokens

        del token_pair_pretokens[pair]
        add_pairs_to_heap((pair, token_pair_counter[pair]) for pair in changed_pairs)



    return vocab, merges

if __name__ == "__main__":
    debug = True
    data_path = "/home/kareemm/src/assignment1-basics/data/TinyStoriesV2-GPT4-" + ("valid.txt" if debug else "train.txt")
    vocab, merges = train_bpe(data_path, 2048, ["<|endoftext|>"])
    with open("vocab.json", "w") as f:
        json.dump({key: value.hex() for key, value in vocab.items()}, f, indent=4)
    with open("merges.json", "w") as f:
        json.dump([[a.hex(), b.hex()] for a, b in merges], f, indent=4)
