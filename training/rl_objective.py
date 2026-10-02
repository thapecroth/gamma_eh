"""Supervised-anchored edit-tag REINFORCE; rewards are supervision, not grammar gold."""
import json
import math
from pathlib import Path

from pairs import hash_file

REWARDS = {"correct_edit": 1., "incorrect_edit": -3., "missed_edit": -1., "damage_clean_token": -5.}


def validate_coefficient(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("RL coefficient must be finite and nonnegative")


def checkpoint_hashes(directory):
    directory = Path(directory)
    paths = sorted(path for path in directory.iterdir() if path.is_file())
    if not (directory / "config.json").is_file() or not any(
            path.name.startswith(("model", "pytorch_model")) and path.suffix in {".safetensors", ".bin"}
            for path in paths):
        raise ValueError("Initial checkpoint requires config and model weights")
    return {path.name: hash_file(path) for path in paths}


def validate_checkpoint_labels(directory, labels):
    """Check the original classifier mapping before from_pretrained can override it."""
    config = json.loads((Path(directory) / "config.json").read_text())
    expected = {str(index): label for index, label in enumerate(labels)}
    if not labels or labels[0] != "KEEP" or len(set(labels)) != len(labels):
        raise ValueError("Training labels must be unique with KEEP at index zero")
    if config.get("id2label") != expected or config.get("label2id") != {
            label: index for index, label in enumerate(labels)}:
        raise ValueError("Initial checkpoint ordered labels do not match the training vocabulary")
    if config.get("model_type") != "bert":
        raise ValueError("Initial checkpoint must be BERT")
    return checkpoint_hashes(directory)


def reward_table(gold, num_labels):
    """One action reward per valid source token; callers first remove ignored gold."""
    import torch
    if num_labels < 1 or gold.ndim != 1 or ((gold < 0) | (gold >= num_labels)).any():
        raise ValueError("Reward labels must be valid nonignored token IDs")
    rewards = torch.full((gold.numel(), num_labels), REWARDS["incorrect_edit"], device=gold.device)
    rewards[:, 0] = REWARDS["missed_edit"]
    rewards[gold == 0] = REWARDS["damage_clean_token"]
    rewards[gold == 0, 0] = 0.
    edits = gold != 0
    rewards[edits, gold[edits]] = REWARDS["correct_edit"]
    return rewards


def policy_gradient_loss(logits, gold, generator):
    """Token-local REINFORCE with detached analytic expected-reward baseline."""
    import torch
    if logits.shape[:-1] != gold.shape or logits.ndim != 3 or logits.shape[-1] < 1:
        raise ValueError("Expected batch x sequence x labels logits and aligned gold")
    if generator is None:
        raise ValueError("RL sampling requires a dedicated generator")
    valid = gold != -100
    if not valid.any():
        return logits.sum() * 0., {"valid_tokens": 0, "sampled_reward": None, "expected_reward": None}
    log_probability = torch.log_softmax(logits[valid], -1)
    probability = log_probability.detach().exp()
    if not torch.isfinite(probability).all():
        raise ValueError("RL logits must be finite on valid tokens")
    rewards = reward_table(gold[valid], logits.shape[-1]).to(dtype=probability.dtype)
    actions = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
    sampled = rewards.gather(1, actions[:, None]).squeeze(-1)
    baseline = (probability * rewards).sum(-1).detach()
    advantage = (sampled - baseline).detach()
    loss = -(advantage * log_probability.gather(1, actions[:, None]).squeeze(-1)).mean()
    return loss, {"valid_tokens": gold[valid].numel(), "sampled_reward": sampled.mean().item(),
                  "expected_reward": baseline.mean().item(), "sampled_edits": (actions != 0).sum().item(),
                  "sampled_clean_damage": ((gold[valid] == 0) & (actions != 0)).sum().item()}


def anchored_loss(logits, gold, weights, coefficient=0., generator=None):
    """Coefficient zero takes the exact existing weighted CE path, without sampling."""
    import torch.nn.functional as functional
    validate_coefficient(coefficient)
    if logits.shape[:-1] != gold.shape or logits.ndim != 3:
        raise ValueError("Expected aligned token logits and gold")
    if not (gold != -100).any():
        return logits.sum() * 0., {"supervised_loss": 0., "policy_gradient_loss": 0., "valid_tokens": 0}
    supervised = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), gold.reshape(-1),
                                         weight=weights, ignore_index=-100)
    details = {"supervised_loss": supervised.detach().item(), "policy_gradient_loss": 0.,
               "valid_tokens": (gold != -100).sum().item()}
    if coefficient == 0:
        return supervised, details
    policy, diagnostics = policy_gradient_loss(logits, gold, generator)
    return supervised + coefficient * policy, {**details, **diagnostics,
                                               "policy_gradient_loss": policy.detach().item()}
