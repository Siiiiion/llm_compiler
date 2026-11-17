"""Model generation shared by production scheduling and structural evaluation."""

import math
import random
from contextlib import contextmanager

import torch

from LLM.core.tokenization_protocol import encode_generation_prompts


def resolve_max_new_tokens(requested, min_gen_tokens=256, gen_token_scale=4.0):
    requested = max(int(requested), 1)
    if min_gen_tokens <= 0 or not math.isfinite(gen_token_scale) or gen_token_scale <= 0:
        raise ValueError("Generation minimum and scale must be positive and finite")
    return max(requested, math.ceil(requested * gen_token_scale), int(min_gen_tokens))


def decode_decision_tokens(tokenizer, token_ids):
    return (
        tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        .strip()
        .split()
    )


@contextmanager
def isolated_generation_rng(seed):
    """Keep deterministic evaluation from changing training's sampling state."""
    python_state = random.getstate()
    import numpy as np

    numpy_state = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def generate_decisions(
    task,
    states,
    input_obj,
    tokenizer,
    model,
    device,
    gen_kwargs,
    trim_last_input_token=False,
    generation_batch_size=8,
    telemetry=None,
    constraint_mode="none",
    constraint_traces=None,
):
    """Use one prompt, context-budget and continuation protocol at every entrypoint."""
    from LLM.data.make_dataset import input_to_tokens

    if not states:
        return []
    prompts = input_to_tokens(task, states, input_obj)
    if not prompts:
        return []
    if constraint_mode not in {"none", "state_v1", "state_v2"}:
        raise ValueError(f"Unknown constraint mode: {constraint_mode}")
    if constraint_mode in {"state_v1", "state_v2"}:
        if trim_last_input_token:
            raise ValueError("State constraints require the complete stage prompt")
        from LLM.generation.constrained_generation import generate_constrained

        return generate_constrained(
            task,
            states,
            prompts,
            tokenizer,
            model,
            device,
            gen_kwargs,
            telemetry,
            constraint_traces,
            **({"constraint_mode": constraint_mode} if constraint_mode == "state_v2" else {}),
        )
    tokenizer.padding_side = "left"
    batch = encode_generation_prompts(tokenizer, prompts)
    responses = []
    context_length = int(tokenizer.model_max_length)
    batch_size = max(int(generation_batch_size), 1)
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            ids = torch.tensor(
                batch["input_ids"][start : start + batch_size], dtype=torch.long, device=device
            )
            mask = torch.tensor(
                batch["attention_mask"][start : start + batch_size], dtype=torch.long, device=device
            )
            if trim_last_input_token and ids.shape[-1] > 1:
                ids, mask = ids[:, :-1], mask[:, :-1]
            kwargs = dict(gen_kwargs)
            available = context_length - ids.shape[-1]
            if available <= 0:
                responses.extend([[] for _ in range(ids.shape[0])])
                if telemetry is not None:
                    telemetry["context_exhausted"] = (
                        telemetry.get("context_exhausted", 0) + ids.shape[0]
                    )
                continue
            kwargs["max_new_tokens"] = min(int(kwargs["max_new_tokens"]), available)
            output = model.generate(input_ids=ids, attention_mask=mask, **kwargs)[
                :, ids.shape[-1] :
            ]
            responses.extend(output.tolist())
            if telemetry is not None:
                telemetry["generation_calls"] = telemetry.get("generation_calls", 0) + 1
                telemetry["token_budget"] = telemetry.get("token_budget", 0) + kwargs[
                    "max_new_tokens"
                ] * len(ids)
                telemetry["generated_tokens"] = telemetry.get("generated_tokens", 0) + int(
                    output.numel()
                )
    return [decode_decision_tokens(tokenizer, response) for response in responses]
