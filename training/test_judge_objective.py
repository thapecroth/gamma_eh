from types import SimpleNamespace

import pytest

from judge_objective import (batch_loss, decode_actions, eligible, reinforce_from_rewards,
                             sample_actions, select_rows, subset_spec)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def synthetic(source="She have a scarf.", **changes):
    return {"source": source, "license": "CC0-1.0", "origin": "original-template-v1", **changes}


def test_subset_is_deterministic_and_excludes_humans_references_contacts_protected():
    rows = [synthetic(f"They have parcel {index}.") for index in range(10)]
    rejected = [synthetic(origin="ErAConD"), synthetic(origin="rule-corruption-v1", seed_origin="ErAConD-human-reference"),
                synthetic(license="MIT"), synthetic(dialog_id="human-dialog"), synthetic("Contact a@example.test."),
                synthetic("Run `items.has(key)` now."), synthetic("Open https://example.test/path.")]
    assert not any(eligible(row) for row in rejected)
    selected = select_rows(rows + rejected, 4, 8)
    assert subset_spec(selected) == subset_spec(select_rows(list(reversed(rows)) + rejected, 4, 8))
    assert eligible(synthetic(origin="rule-corruption-v1", seed_origin="agent-authored-rule-seeds-v1"))


def test_padding_excluded_sequence_logprob_summed_and_rng_isolated(torch):
    logits = torch.tensor([[[.1, .3], [.2, .5], [100., -100.]]], requires_grad=True)
    valid = torch.tensor([[True, True, False]])
    state = torch.random.get_rng_state().clone()
    sampled, greedy, log_probability = sample_actions(logits, valid, torch.Generator().manual_seed(4))
    expected = logits.log_softmax(-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)[:, :2].sum(-1)
    torch.testing.assert_close(log_probability, expected)
    assert sampled[0, 2] == greedy[0, 2] == 0
    assert torch.equal(torch.random.get_rng_state(), state)
    log_probability.sum().backward()
    assert torch.equal(logits.grad[0, 2], torch.zeros(2))
    for invalid, mask in ((logits.detach() * float("nan"), valid), (logits, valid[:, :2])):
        with pytest.raises(ValueError): sample_actions(invalid, mask, torch.Generator().manual_seed(4))


def test_sentence_reward_gradient_and_detached_baseline(torch):
    log_probability = torch.tensor([-2., -3.], requires_grad=True)
    baseline = torch.tensor([.4, .8], requires_grad=True)
    loss, metrics = reinforce_from_rewards(log_probability, [.8, .2], baseline)
    loss.backward()
    torch.testing.assert_close(log_probability.grad, torch.tensor([-.2, .3]))
    assert baseline.grad is None and metrics["nonzero_advantages"] == 2
    with pytest.raises(ValueError): reinforce_from_rewards(log_probability, [float("nan"), .2], baseline)


def test_decode_uses_selected_actions_original_offsets_and_guards(torch):
    rows = [synthetic("She have a scarf.")]
    words = [[{"text": text, "start": start, "end": end} for text, start, end in
              (("She", 0, 3), ("have", 4, 8), ("a", 9, 10), ("scarf", 11, 16), (".", 16, 17))]]
    positions = [[(index, index) for index in range(5)]]
    actions = torch.tensor([[0, 1, 0, 0, 0]])
    probability = torch.full((1, 5, 2), .5)
    assert decode_actions(rows, words, positions, actions, probability, ["KEEP", "REPLACE:has"], 2) == ["She has a scarf."]
    assert decode_actions(rows, words, positions, actions, probability, ["KEEP", "REPLACE:are"], 2) == [rows[0]["source"]]


def test_eval_forward_restores_global_rng_and_training_mode(torch, monkeypatch):
    import judge_objective
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.tensor([.1, .3]))
        def forward(self, **feeds):
            assert not self.training
            torch.rand(3)
            return SimpleNamespace(logits=self.parameter.reshape(1, 1, 2).expand(1, 3, 2))
    feeds = {name: torch.zeros((1, 3), dtype=torch.long) for name in ("input_ids", "attention_mask", "token_type_ids")}
    monkeypatch.setattr(judge_objective, "encode_rows", lambda *unused: (feeds, torch.tensor([[False, True, False]]), [[]], [[]]))
    monkeypatch.setattr(judge_objective, "decode_actions", lambda *unused: ["She has a scarf."])
    judge = SimpleNamespace(rewards=lambda pairs: [.8, .4])
    model = Model()
    state = torch.random.get_rng_state().clone()
    loss, stats = batch_loss(model, [synthetic()], None, ["KEEP", "REPLACE:has"], 2, 96, "cpu",
                             torch.Generator().manual_seed(4), judge)
    assert model.training and torch.equal(torch.random.get_rng_state(), state)
    loss.backward()
    assert model.parameter.grad is not None and stats["nonzero_advantages"] == 1
