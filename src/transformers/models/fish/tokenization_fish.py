"""Tokenization classes for Fish Speech"""

import base64
import json
from pathlib import Path
import tiktoken

from typing import Union, List

from ...tokenization_utils import PreTrainedTokenizer

FISH_TIKTOKEN_PATTERN = "|".join(
    [
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        r"\p{P}",
        r"[^\r\n\p{L}\p{N}]?\p{L}+",
        r"\p{N}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(\?!\S)",
        r"\s+",
    ]
)
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
PAD_TOKEN = "<|pad|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

MODALITY_TEXT_TOKEN = "<|text|>"
MODALITY_VOICE_TOKEN = "<|voice|>"
MODALITY_INTERLEAVE_TOKEN = "<|interleave|>"
MODALITY_TOKENS = {
    "text": MODALITY_TEXT_TOKEN,
    "voice": MODALITY_VOICE_TOKEN,
    "interleave": MODALITY_INTERLEAVE_TOKEN,
}

PLACEHOLDER_TOKEN = [""] * 4
for i in range(4):
    PLACEHOLDER_TOKEN[i] = f"<|placeholder:{i}|>"

SEMANTIC_TOKEN_TEMPLATE = "<|semantic:{i}|>"
SEMANTIC_TOKENS = [SEMANTIC_TOKEN_TEMPLATE.format(i=i) for i in range(1024)]

# Warning: when you add a new special token, you should only add it to the end of the list.
ALL_SPECIAL_TOKENS = [
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    IM_START_TOKEN,
    IM_END_TOKEN,
    PLACEHOLDER_TOKEN[0],
    PLACEHOLDER_TOKEN[1],
    PLACEHOLDER_TOKEN[2],
    PLACEHOLDER_TOKEN[3],
    MODALITY_TEXT_TOKEN,
    MODALITY_VOICE_TOKEN,
    MODALITY_INTERLEAVE_TOKEN,
    *SEMANTIC_TOKENS,
]

class FishTokenizer(PreTrainedTokenizer)
    def __init__(self, model_path: str) -> None:
        mergeable_ranks = self.load_tiktoken_bpe(model_path)
        special_token_begin = len(mergeable_ranks)
        self.all_special_tokens_with_ids = {
            token: special_token_begin + i for i, token in enumerate(ALL_SPECIAL_TOKENS)
        }
        self.semantic_id_to_token_id = {
            i: self.all_special_tokens_with_ids[token]
            for i, token in enumerate(SEMANTIC_TOKENS)
        }
        self.semantic_begin_id = self.all_special_tokens_with_ids[SEMANTIC_TOKENS[0]]
        self.semantic_end_id = self.all_special_tokens_with_ids[SEMANTIC_TOKENS[-1]]

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=FISH_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.all_special_tokens_with_ids,
        )

    def load_tiktoken_bpe(tiktoken_bpe_file: str) -> dict[bytes, int]:
        data = {}
        for line in open(tiktoken_bpe_file).read().splitlines():
            if not line:
                continue
            token, rank = line.split()
            data[base64.b64decode(token)] = int(rank)
        return data

    def get_token_id(self, token: str) -> int:
        return self.all_special_tokens_with_ids[token]

    def tokenize(text: str) -> List[str]:
        subs = []
        for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS):
            subs.append(text[i : i + TIKTOKEN_MAX_ENCODE_CHARS])
        return subs

    def convert_tokens_to_ids(self, tokens: List[str], allowed_special: bool | set[str] = True) -> List[int]:
        if allowed_special is True:
            allowed_special = self.tkt_model.special_tokens_set
        elif allowed_special is False:
            allowed_special = set()

        return sum(
            self.tkt_model.encode_batch(
                subs, allowed_special=allowed_special, disallowed_special=set()
            ),
            start=[],
        )

    def encode(self, text: str | list, allowed_special: bool | set[str] = True) -> list[int]:
        if isinstance(text, str):
            tokens = self.tokenize(text)
            ids = self.convert_tokens_to_ids(tokens, allowed_special)
        elif isinstance(text, list):
            ids = self.convert_tokens_to_ids(text, allowed_special)
        else:
            assert isinstance(text, str) and isinstance(text, list)
            
        return ids

    def decode(self, tokens: list[int]) -> str:
        return self.tkt_model.decode(tokens)

    def save_pretrained(self, save_directory: str):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "tokenizer.tiktoken", "w") as f:
            for token, rank in self.tkt_model._mergeable_ranks.items():
                f.write(f"{base64.b64encode(token).decode()} {rank}\n")

        with open(path / "special_tokens.json", "w") as f:
            json.dump(
                self.all_special_tokens_with_ids,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def from_pretrained(pretrained_model_name_or_path: Union[str, os.PathLike]):
        path = Path(pretrained_model_name_or_path)
        if os.path.isfile(path):
            return FishTokenizer(path)
        else:
            return FishTokenizer(path / "tokenizer.tiktoken")
