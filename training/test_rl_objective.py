import json

import pytest

from rl_objective import (anchored_loss, policy_gradient_loss, reward_table,
                          validate_checkpoint_labels, validate_coefficient)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_coefficient_rejects_nonfinite_and_negative_values():
    for value in (-1., float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            validate_coefficient(value)
    validate_coefficient(0.)


def test_checkpoint_rejects_same_size_reordered_labels_before_loading(tmp_path):
    config = {"model_type": "bert", "id2label": {"0": "KEEP", "1": "DELETE", "2": "REPLACE:has"},
              "label2id": {"KEEP": 0, "DELETE": 1, "REPLACE:has": 2}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"fixture model")
    with pytest.raises(ValueError, match="ordered labels"):
        validate_checkpoint_labels(tmp_path, ["KEEP", "REPLACE:has", "DELETE"])
    hashes = validate_checkpoint_labels(tmp_path, ["KEEP", "DELETE", "REPLACE:has"])
    assert set(hashes) == {"config.json", "model.safetensors"}
    config["label2id"]["DELETE"] = 2
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="ordered labels"):
        validate_checkpoint_labels(tmp_path, ["KEEP", "DELETE", "REPLACE:has"])


def test_clean_damage_and_wrong_edits_cost_more_than_missed_edits(torch):
    rewards = reward_table(torch.tensor([0, 1]), 3)
    torch.testing.assert_close(rewards, torch.tensor([[0., -5., -5.], [-1., 1., -3.]]))
    with pytest.raises(ValueError, match="nonignored"):
        reward_table(torch.tensor([-100]), 3)


def test_zero_coefficient_is_exact_weighted_ce_and_does_not_consume_rng(torch):
    logits = torch.tensor([[[.1, .8], [.5, .4], [.3, .2]]], requires_grad=True)
    gold = torch.tensor([[1, 0, -100]])
    weights = torch.tensor([.3, 1.])
    generator = torch.Generator().manual_seed(42)
    rng = generator.get_state().clone()
    loss, details = anchored_loss(logits, gold, weights, 0., generator)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), gold.reshape(-1),
                                                weight=weights, ignore_index=-100)
    assert torch.equal(loss, expected)
    assert torch.equal(generator.get_state(), rng)
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], torch.autograd.grad(expected, logits)[0])
    assert details["policy_gradient_loss"] == 0.


def test_sampling_is_reproducible_without_changing_global_rng(torch):
    logits = torch.zeros((1, 80, 3), requires_grad=True)
    gold = torch.zeros((1, 80), dtype=torch.long)
    global_rng = torch.random.get_rng_state().clone()
    first, a = policy_gradient_loss(logits, gold, torch.Generator().manual_seed(8))
    second, b = policy_gradient_loss(logits, gold, torch.Generator().manual_seed(8))
    assert torch.equal(first, second) and a == b
    assert torch.equal(torch.random.get_rng_state(), global_rng)


def test_ignored_positions_do_not_change_sampling_reward_or_gradient(torch):
    logits = torch.tensor([[[.3, .8, .2], [.1, .2, .7]]], requires_grad=True)
    gold = torch.tensor([[1, 0]])
    padded = torch.cat([logits.detach(), torch.tensor([[[100., -90., 20.]]])], dim=1).requires_grad_()
    padded_gold = torch.tensor([[1, 0, -100]])
    a, details_a = anchored_loss(logits, gold, torch.ones(3), .1, torch.Generator().manual_seed(9))
    b, details_b = anchored_loss(padded, padded_gold, torch.ones(3), .1, torch.Generator().manual_seed(9))
    torch.testing.assert_close(a, b)
    assert details_a == details_b
    gradient = torch.autograd.grad(b, padded)[0]
    assert torch.equal(gradient[:, -1], torch.zeros((1, 3)))
    torch.testing.assert_close(torch.autograd.grad(a, logits)[0], gradient[:, :2])


def test_all_ignored_rows_return_zero_and_do_not_sample(torch):
    logits = torch.ones((1, 4, 2), requires_grad=True)
    generator = torch.Generator().manual_seed(4)
    state = generator.get_state().clone()
    loss, details = anchored_loss(logits, torch.full((1, 4), -100), torch.ones(2), .1, generator)
    loss.backward()
    assert loss.item() == 0. and details["valid_tokens"] == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    assert torch.equal(generator.get_state(), state)


def test_detached_analytic_baseline_preserves_expected_reward_gradient(torch, monkeypatch):
    logits = torch.tensor([[[.2, .5, -.1]]], requires_grad=True)
    gold = torch.tensor([[1]])
    probability = logits.detach().softmax(-1).reshape(-1)
    gradient = torch.zeros_like(logits)
    for action in range(3):
        monkeypatch.setattr(torch, "multinomial", lambda value, count, generator, selected=action:
                            torch.tensor([[selected]], device=value.device))
        loss, details = policy_gradient_loss(logits, gold, torch.Generator().manual_seed(4))
        gradient += probability[action] * torch.autograd.grad(loss, logits)[0]
    expected = (logits.softmax(-1) * torch.tensor([-1., 1., -3.])).sum()
    assert details["expected_reward"] == pytest.approx(expected.item())
    torch.testing.assert_close(gradient, -torch.autograd.grad(expected, logits)[0])
