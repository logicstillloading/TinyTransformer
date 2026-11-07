"""
16k English-focused BPE Tokenizer trained from scratch on the pretraining dataset mixture.
Uses ByteLevel pre-tokenization for 100% vocabulary coverage (zero out-of-vocabulary crashes).
"""

import os
from typing import List, Optional, Iterator
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


TOOL_AND_AGENT_SPECIAL_TOKENS = [
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<thought>", "</thought>",
    "<system>", "</system>",
    "<user>", "</user>",
    "<assistant>", "</assistant>",
]
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"] + TOOL_AND_AGENT_SPECIAL_TOKENS
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
DEFAULT_VOCAB_SIZE = 16396


class Tokenizer:
    def __init__(self, tokenizer: HFTokenizer):
        self._tokenizer = tokenizer
        self.pad_token_id = PAD_ID
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID
        self.unk_token_id = UNK_ID

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        ids = self._tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._tokenizer.save(path)

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        hf_tok = HFTokenizer.from_file(path)
        existing = hf_tok.get_vocab()
        missing = [t for t in TOOL_AND_AGENT_SPECIAL_TOKENS if t not in existing]
        if missing:
            hf_tok.add_special_tokens(missing)
        return cls(hf_tok)


def train_tokenizer_from_iterator(
    text_iterator: Iterator[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    save_path: Optional[str] = "tokenizer.json",
) -> Tokenizer:
    """
    Trains a 16k ByteLevel BPE tokenizer from an iterator of raw texts.
    """
    tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    wrapped = Tokenizer(tokenizer)
    if save_path:
        wrapped.save(save_path)
    return wrapped


def get_default_training_corpus() -> Iterator[str]:
    """~60k docs sampled in the same proportions as the pretraining mixture."""
    from datasets import load_dataset

    streams = [
        ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 21000),
        ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", 18000),
        ("HuggingFaceTB/finemath", "finemath-4plus", 9000),
        ("Avelina/python-edu-cleaned", None, 6000),
        ("wikimedia/wikipedia", "20231101.en", 3000),
        ("HuggingFaceTB/smol-smoltalk", None, 3000),
    ]
    for ds_name, config_name, max_samples in streams:
        n = 0
        try:
            print(f"[Tokenizer] Streaming {max_samples} samples from {ds_name} ({config_name})...")
            if config_name is not None:
                ds = load_dataset(ds_name, config_name, split="train", streaming=True)
            else:
                ds = load_dataset(ds_name, split="train", streaming=True)
            ds = ds.shuffle(seed=0, buffer_size=10_000)
            for item in ds:
                if n >= max_samples:
                    break
                if "messages" in item:
                    text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in item["messages"])
                else:
                    text = item.get("text", "")
                if text and len(text.strip()) > 50:
                    n += 1
                    yield text[:12000]  # cap so long docs don't dominate merges
        except Exception as e:
            raise RuntimeError(f"[Tokenizer] stream {ds_name} failed after {n} docs: {e}")
        print(f"[Tokenizer]   -> {n} docs")


def get_or_create_tokenizer(path: str = "tokenizer.json", vocab_size: int = DEFAULT_VOCAB_SIZE) -> Tokenizer:
    """Loads existing tokenizer or trains a fresh one."""
    if os.path.exists(path):
        print(f"Loading existing tokenizer from {path}")
        return Tokenizer.from_file(path)

    print(f"Training new 16k BPE tokenizer from dataset corpus...")
    corpus = get_default_training_corpus()
    tokenizer = train_tokenizer_from_iterator(corpus, vocab_size=vocab_size, save_path=path)
    print(f"Tokenizer trained with vocab_size={tokenizer.vocab_size} and saved to {path}")
    return tokenizer
