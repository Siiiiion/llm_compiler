"""Shared tokenization rules for schedule prompts and decision suffixes.

The schedule format is whitespace separated, but the Qwen tokenizer has a
real token for a separating space in several contexts.  Encoding a prompt and
suffix independently therefore does not necessarily produce the same ids as
encoding the text that is presented to the model.  Keep the boundary protocol
in one small module so training, generation and structural evaluation cannot
silently drift apart.
"""

from typing import Any, Dict, List, Sequence, Tuple


PPT_BOUNDARY_VERSION = "ppt_boundary_v2"


def build_ppt_marker_id_candidates(tokenizer, marker: str = "PPT") -> List[List[int]]:
    """Return marker patterns whose end is the marker, never its trailing space."""
    candidates: List[List[int]] = []
    # A leading separator may be a token of its own.  It is part of the
    # matching context, but the returned end index still points at PPT.
    for text in (marker, f" {marker}"):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids and ids not in candidates:
            candidates.append(ids)
    candidates.sort(key=len, reverse=True)
    return candidates


def find_ppt_marker_end(input_ids: Sequence[int], patterns: Sequence[Sequence[int]]) -> int:
    """Return the inclusive id index of PPT, excluding a following separator."""
    for start in range(len(input_ids)):
        for pattern in patterns:
            end = start + len(pattern)
            if end <= len(input_ids) and list(input_ids[start:end]) == list(pattern):
                return end - 1
    return -1


def encode_generation_prompts(tokenizer, prompts: Sequence[str]) -> Dict[str, Any]:
    """Encode prompts exactly as model conditions (without implicit specials)."""
    return tokenizer(
        list(prompts),
        add_special_tokens=False,
        padding=True,
        max_length=None,
    )


def encode_prompt_suffix(
    tokenizer,
    prompt: str,
    suffix: str,
    append_eos: bool = False,
) -> Tuple[List[int], List[int], List[int]]:
    """Encode a prompt and suffix as one text and return ids plus label mask.

    The prompt ends at the final character of ``prompt`` (normally ``PPT`` or
    another stage marker).  The separating space is deliberately included in
    the supervised suffix.  A prefix assertion prevents a tokenizer upgrade
    from silently making the training condition differ from generation.
    """
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    # Dataset builders emit whitespace-separated fields.  Avoid introducing a
    # second separator if a hand-authored suffix already starts with whitespace.
    separator = "" if not suffix or suffix[0].isspace() else " "
    combined_ids = list(
        tokenizer(prompt + separator + suffix, add_special_tokens=False)["input_ids"]
    )
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Tokenizer changed prompt ids when suffix was appended; rebuild the "
            "dataset or use a tokenizer with stable prompt-prefix encoding."
        )
    suffix_ids = combined_ids[len(prompt_ids) :]
    if append_eos and tokenizer.eos_token_id is not None:
        suffix_ids = list(suffix_ids) + [int(tokenizer.eos_token_id)]
        combined_ids = list(prompt_ids) + suffix_ids
    labels = [-100] * len(prompt_ids) + suffix_ids
    return prompt_ids, combined_ids, labels
