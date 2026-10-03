"""Synthetic-only, guarded sentence REINFORCE using independent LLM verdicts."""
import hashlib
import math
import re

from data import TOKEN_RE
from evaluate import apply_proposals, decode_proposal, protected_spans
from judge import digest

CONTACT = re.compile(r"https?://|\b[^\s@]+@[^\s@]+\.[^\s@]+|\b(?:sk-|AKIA)[A-Za-z0-9]{12,}", re.I)
SEED_ORIGIN = "agent-authored-rule-seeds-v1"
ONLINE_ORIGIN = "cliproxy-original-context-v1"


def online_provenance(row):
    fingerprint = row.get("generator_spec_sha256")
    return (row.get("origin") == ONLINE_ORIGIN and row.get("license") == "provider-terms-unverified"
            and row.get("machine_generated_original") is True and row.get("human_reviewed") is False
            and row.get("publication_allowed") is False and row.get("generation_model") == "gpt-6-luna"
            and isinstance(row.get("generation_request_id"), str) and bool(row["generation_request_id"])
            and isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None)


def eligible(row):
    original = row.get("license") == "CC0-1.0" and (row.get("origin") == "original-template-v1" or (
        row.get("origin") == "rule-corruption-v1" and row.get("seed_origin") == SEED_ORIGIN))
    source = row.get("source")
    return ((original or online_provenance(row)) and not row.get("dialog_id")
            and isinstance(source, str) and 3 <= len(source) <= 600 and source.isascii()
            and not CONTACT.search(source) and not protected_spans(source)
            and not any(ord(character) < 32 for character in source))


def select_rows(rows, limit=128, seed=42):
    if not 1 <= limit <= 256: raise ValueError("RL subset must contain at most 256 rows per epoch")
    unique = {row["source"]: row for row in rows if eligible(row)}
    ordered = sorted(unique, key=lambda source: hashlib.sha256(f"{seed}:{source}".encode()).hexdigest())
    selected = [unique[source] for source in ordered[:limit]]
    if not selected: raise ValueError("No eligible original synthetic RL sources")
    return selected


def subset_spec(rows):
    return {"rows": len(rows), "source_sha256": [digest(row["source"]) for row in rows],
            "origins": sorted({row["origin"] for row in rows}),
            "scope": "Original synthetic sources only: CC0 templates/seeds or provenance-recorded generated contexts; human corrections/references never sent to judge."}


def encode_rows(rows, tokenizer, max_length):
    if not rows or not 4 <= max_length <= 512 or any(not eligible(row) for row in rows):
        raise ValueError("Invalid RL source batch or context")
    words = [[{"text": match.group(), "start": match.start(), "end": match.end()}
              for match in TOKEN_RE.finditer(row["source"])] for row in rows]
    if any(not value for value in words): raise ValueError("Empty RL token sequence")
    encoded = tokenizer([[word["text"] for word in value] for value in words], is_split_into_words=True,
                        padding=True, return_tensors="pt")
    if encoded["input_ids"].shape[1] > max_length: raise ValueError("Frozen RL source exceeds context budget")
    import torch
    valid = torch.zeros_like(encoded["input_ids"], dtype=torch.bool)
    positions = []
    for index, value in enumerate(words):
        previous, word_positions = None, []
        for position, word_index in enumerate(encoded.word_ids(index)):
            if word_index is not None and word_index != previous:
                valid[index, position] = True
                word_positions.append((position, word_index))
            previous = word_index
        if {word_index for _, word_index in word_positions} != set(range(len(value))):
            raise ValueError("RL tokenizer did not represent every source word")
        positions.append(word_positions)
    return {name: encoded[name] for name in ("input_ids", "attention_mask", "token_type_ids")}, valid, words, positions


def sample_actions(logits, valid, generator, temperature=1.):
    import torch
    if not math.isfinite(temperature) or temperature <= 0 or logits.ndim != 3 or logits.shape[:-1] != valid.shape:
        raise ValueError("Invalid RL action dimensions or temperature")
    if valid.dtype != torch.bool: raise ValueError("RL valid-token mask must be boolean")
    if not valid.any(dim=1).all() or logits.shape[-1] < 1 or not torch.isfinite(logits).all():
        raise ValueError("RL rows require finite logits and valid source words")
    if generator is None: raise ValueError("RL requires a dedicated sampling generator")
    log_probability = torch.log_softmax(logits / temperature, -1)
    sampled = torch.zeros_like(valid, dtype=torch.long)
    sampled[valid] = torch.multinomial(log_probability[valid].detach().exp(), 1, generator=generator).squeeze(-1)
    greedy = logits.argmax(-1).masked_fill(~valid, 0)
    sequence_log_probability = log_probability.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).masked_fill(~valid, 0.).sum(-1)
    return sampled, greedy, sequence_log_probability


def decode_actions(rows, words, positions, actions, probabilities, labels, schema):
    import torch
    if (len(rows) != len(words) or len(rows) != len(positions) or actions.ndim != 2 or actions.shape[0] != len(rows)
            or probabilities.ndim != 3 or probabilities.shape[:2] != actions.shape
            or probabilities.shape[-1] != len(labels) or actions.dtype != torch.long
            or not torch.isfinite(probabilities).all()):
        raise ValueError("RL source/action batch mismatch")
    if not labels or labels[0] != "KEEP": raise ValueError("RL requires KEEP at label zero")
    candidates = []
    for index, row in enumerate(rows):
        proposals = []
        if (len({position for position, _ in positions[index]}) != len(positions[index])
                or any(not 0 <= position < actions.shape[1] or not 0 <= word_index < len(words[index])
                       for position, word_index in positions[index])):
            raise ValueError("Invalid RL word/action positions")
        for position, word_index in positions[index]:
            action = int(actions[index, position])
            if not 0 <= action < len(labels): raise ValueError("RL action outside edit vocabulary")
            confidence = float(probabilities[index, position, action])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1: raise ValueError("Invalid RL action probability")
            proposal = decode_proposal(row["source"], words[index], word_index, labels[action], confidence, schema)
            if proposal: proposals.append(proposal)
        candidate = apply_proposals(row["source"], proposals, 0.)
        if not candidate.strip() or len(candidate) > 2000 or CONTACT.search(candidate):
            raise ValueError("Unsafe or unbounded RL candidate")
        candidates.append(candidate)
    return candidates


def reinforce_from_rewards(log_probability, rewards, baseline):
    import torch
    rewards = torch.as_tensor(rewards, dtype=log_probability.dtype, device=log_probability.device)
    baseline = torch.as_tensor(baseline, dtype=log_probability.dtype, device=log_probability.device)
    if log_probability.ndim != 1 or rewards.shape != log_probability.shape or baseline.shape != rewards.shape:
        raise ValueError("Sentence reward dimensions mismatch")
    if not all(torch.isfinite(value).all() for value in (log_probability, rewards, baseline)) or (
            ((rewards < 0) | (rewards > 1) | (baseline < 0) | (baseline > 1)).any()):
        raise ValueError("Sentence rewards/log probabilities must be finite with rewards in [0,1]")
    advantage = (rewards - baseline).detach()
    return -(advantage * log_probability).mean(), {
        "rows": rewards.numel(), "reward": rewards.mean().item(), "baseline_reward": baseline.mean().item(),
        "advantage_abs": advantage.abs().mean().item(), "nonzero_advantages": (advantage != 0).sum().item()}


def batch_loss(model, rows, tokenizer, labels, schema, max_length, device, generator, judge, temperature=1.):
    import torch
    feeds, valid, words, positions = encode_rows(rows, tokenizer, max_length)
    feeds, valid = {name: value.to(device) for name, value in feeds.items()}, valid.to(device)
    device_value = torch.device(device)
    devices = [device_value.index if device_value.index is not None else torch.cuda.current_device()] if device_value.type == "cuda" else []
    training = model.training
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            logits = model(**feeds).logits
            sampled, greedy, log_probability = sample_actions(logits, valid, generator, temperature)
            probability = (logits.detach() / temperature).softmax(-1).cpu()
            sampled_text = decode_actions(rows, words, positions, sampled.cpu(), probability, labels, schema)
            baseline_text = decode_actions(rows, words, positions, greedy.cpu(), probability, labels, schema)
            pairs = [(row["source"], candidate) for row, sample, baseline in zip(rows, sampled_text, baseline_text)
                     for candidate in (sample, baseline)]
            scores = judge.rewards(pairs)
            if len(scores) != len(rows) * 2: raise ValueError("Judge sentence rewards incomplete")
            loss, stats = reinforce_from_rewards(log_probability, scores[::2], scores[1::2])
            return loss, {**stats, "sampled_changed": sum(row["source"] != value for row, value in zip(rows, sampled_text)),
                          "greedy_changed": sum(row["source"] != value for row, value in zip(rows, baseline_text))}
    finally:
        model.train(training)
