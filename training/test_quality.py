"""Pure-stdlib quality workflow boundaries; these also run without training dependencies."""
import json
from pathlib import Path
from argparse import Namespace

import pytest

from data import edit_tags, tokens
from edit_ops import decode_word, reconstruct
from evaluate import (apply_proposals, choose_calibration, choose_joint_calibration, decode_proposal,
                      evaluate_records, score_predictions)
from experiments import build, common_training_population, select_rows
from import_jfleg import materialize
from pairs import evaluation_keys, group_id
from prepare_pairs import prepare


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def screened_pair(**overrides):
    from teacher_recipes import lexical_family
    from verified_teacher import CRITIC_FIELDS, blind_id, critic_id
    target = "She is ready for the workshop."
    row = {"source": "She are ready for the workshop.", "target": target, "category": "agreement",
           "origin": "glm-recipe-machine-screened", "model": "glm-5.3",
           "license": "provider-terms-unverified", "status": "screened", "admission_status": "screened",
           "review_status": "machine-verified", "split": "train", "human_reviewed": False,
           "weak_supervision": True, "family_id": lexical_family(target), "recipe_id": "agreement_subject",
           "recipe_version": 1, "run_fingerprint": "a" * 64, "row_id": "family-00000000:variant",
           "guard_coverage": {"alignment": True, "guarded": True}}
    row["blind_review"] = {"id": blind_id(row, row["run_fingerprint"]), "corrected": target, "source_is_grammatical": False}
    row["pair_review"] = {"id": row["row_id"], **dict.fromkeys(CRITIC_FIELDS, True)}
    row["critic_request_id"] = critic_id(row, row["run_fingerprint"])
    return {**row, **overrides}


def test_preparation_retains_machine_provenance_without_promoting_review_or_license(tmp_path):
    source = tmp_path / "pairs.jsonl"
    write_rows(source, [screened_pair()])
    output = tmp_path / "prepared"
    manifest = prepare([source], output, allow_weak_train=True, allow_unverified_teacher_terms=True, schema=2)
    row = json.loads((output / "train.jsonl").read_text())
    assert row["review_status"] == "weak-supervision" and row["human_reviewed"] is False
    assert row["recipe_version"] == 1 and row["run_fingerprint"] == "a" * 64
    assert row["clean_group"] == screened_pair()["family_id"]
    assert row["blind_review"] == screened_pair()["blind_review"]
    assert row["critic_request_id"] == screened_pair()["critic_request_id"]
    assert manifest["publication_allowed"] is False
    assert manifest["evaluation"]["dev"]["rows"] == manifest["evaluation"]["test"]["rows"] == 0


@pytest.mark.parametrize("overrides,reason", [
    ({"admission_status": "quarantined", "status": "screened"}, "not_admitted"),
    ({"split": "dev"}, "teacher_heldout"),
    ({"split": "test"}, "teacher_heldout"),
    ({"blind_review": None}, "teacher_evidence_missing"),
    ({"human_reviewed": True}, "teacher_evidence_missing"),
    ({"pair_review": {"target_is_grammatical": True}}, "teacher_evidence_missing"),
    ({"critic_request_id": "wrong-wire-id"}, "teacher_evidence_missing"),
])
def test_preparation_rejects_revocations_heldout_teacher_families_and_missing_evidence(tmp_path, overrides, reason):
    source = tmp_path / "pairs.jsonl"
    write_rows(source, [screened_pair(**overrides)])
    output = tmp_path / "prepared"
    manifest = prepare([source], output, allow_weak_train=True, allow_unverified_teacher_terms=True, schema=2)
    assert not (output / "train.jsonl").read_text()
    assert manifest["counts"]["rejected:" + reason] == 1


@pytest.mark.parametrize("field,value", [("meaning_preserved", False), ("minimal_edit", "true"), ("id", "wrong-row")])
def test_preparation_rechecks_explicit_successful_critic_verdicts(tmp_path, field, value):
    row = screened_pair()
    row["pair_review"][field] = value
    source = tmp_path / "pairs.jsonl"
    write_rows(source, [row])
    manifest = prepare([source], tmp_path / "prepared", allow_weak_train=True, allow_unverified_teacher_terms=True, schema=2)
    assert manifest["splits"]["train"]["accepted"] == 0
    assert manifest["counts"]["rejected:teacher_evidence_missing"] == 1


def test_preparation_rechecks_exact_blind_repair_target(tmp_path):
    row = screened_pair()
    row["blind_review"]["corrected"] = "She was ready for the workshop."
    source = tmp_path / "pairs.jsonl"
    write_rows(source, [row])
    manifest = prepare([source], tmp_path / "prepared", allow_weak_train=True, allow_unverified_teacher_terms=True, schema=2)
    assert manifest["splits"]["train"]["accepted"] == 0


@pytest.mark.parametrize("text", ["We use --help to inspect the CLI.", "Mira arrived—then the meeting began.",
                                  "The folder  is ready for review."])
def test_preparation_retains_exact_clean_keep_controls_and_original_text(tmp_path, text):
    from teacher_recipes import lexical_family
    row = screened_pair(source=text, target=text, category="clean", family_id=lexical_family(text))
    row["blind_review"].update(corrected=text, source_is_grammatical=True)
    row["pair_review"].update(source_has_error=False, correction_is_necessary=False)
    source = tmp_path / "pairs.jsonl"
    write_rows(source, [row])
    output = tmp_path / "prepared"
    manifest = prepare([source], output, allow_weak_train=True, allow_unverified_teacher_terms=True, schema=2)
    prepared = json.loads((output / "train.jsonl").read_text())
    assert prepared["source"] == prepared["target"] == text
    assert prepared["tags"] and set(prepared["tags"]) == {"KEEP"}
    assert prepared["review_status"] == "weak-supervision" and prepared["human_reviewed"] is False
    assert manifest["splits"]["train"]["accepted"] == 1
    assert manifest["counts"].get("rejected", 0) == 0


def words(text):
    from edit_ops import TOKEN_RE
    return [{"text": match.group(), "start": match.start(), "end": match.end()} for match in TOKEN_RE.finditer(text)]


def test_shared_legacy_guard_fixture():
    fixture = json.loads(Path(__file__).with_name("legacy-guard-parity.json").read_text())
    for row in fixture["cases"]:
        tokens = words(row["source"])
        index = [i for i, token in enumerate(tokens) if token["text"] == row["anchor"]][row.get("occurrence", 0)]
        proposal = decode_proposal(row["source"], tokens, index, row["tag"], .999, 1)
        result = apply_proposals(row["source"], [proposal] if proposal else [], .85)
        assert result == row["target"], row


def test_shared_decoder_fixture_and_invalid_payloads():
    fixture = json.loads(Path(__file__).with_name("edit-parity.json").read_text())
    for case in fixture["cases"]:
        assert decode_word(case["word"], case["tag"], 2) == case["tokens"]
    for case in fixture["invalid"]:
        with pytest.raises(ValueError): decode_word(case["word"], case["tag"], 2)
    with pytest.raises(ValueError): decode_word("word", "CASE:UPPER", 1)


@pytest.mark.parametrize("source,target", [
    ("Book.", "A book."), ("He study here.", "He studies here."),
    ("We need book.", "We need a small book."), ("hello world.", "Hello, world!"),
    ("She reads.", "The careful teacher reads."), ("HELLO.", "hello."),
    ("We go home.", "We will be going home."),
])
def test_v2_alignment_roundtrip(source, target):
    source_words, tags = edit_tags(source, target, 2)
    assert reconstruct(source_words, tags, 2) == target
    assert source_words == tokens(source)


def test_v1_tags_remain_default_and_v2_payloads_are_bounded():
    assert edit_tags("He work here.", "He works here.")[1][1] == "REPLACE:works"
    assert edit_tags("He work here.", "He works here.", 2)[1][1] == "SUFFIX:ADD_S"
    with pytest.raises(ValueError): edit_tags("Book.", "One two three four five book.", 2)
    with pytest.raises(ValueError): reconstruct(["word"], [], 2)


def test_prepare_retains_unsupported_reviewed_targets_and_prevents_leakage(tmp_path):
    target = next(f"The careful teacher reads for task {index}." for index in range(1000)
                  if int(group_id(f"The careful teacher reads for task {index}.")[:8], 16) % 100 >= 90)
    source = tmp_path / "pairs.jsonl"
    write_rows(source, [
        {"source": "She reads.", "target": target, "category": "mixed", "license": "CC0-1.0", "review_status": "human-reviewed"},
        {"source": target.replace("reads", "read"), "target": target, "category": "agreement", "license": "CC0-1.0", "review_status": "unreviewed"},
        {"source": "He have a pen.", "target": "He has a pen.", "category": "agreement", "license": "CC0-1.0", "review_status": "unreviewed"},
    ])
    output = tmp_path / "prepared"
    manifest = prepare([source], output, allow_weak_train=True)
    assert manifest["evaluation"]["test"]["rows"] == 1
    assert manifest["splits"]["test"]["accepted"] == 0
    assert manifest["coverage_by_category"]["test"]["mixed"]["unrepresentable"] == 1
    assert manifest["counts"]["weak_heldout_group_dropped"] == 1
    population = json.loads((output / "evaluation/test.jsonl").read_text())
    assert population["source"] == "She reads."
    assert population["references"] == [target]
    assert target not in (output / "train.jsonl").read_text()


def test_prepare_keeps_heldout_rows_outside_train_vocabulary(tmp_path):
    target = next(f"They scrutinize project {index}." for index in range(1000)
                  if int(group_id(f"They scrutinize project {index}.")[:8], 16) % 100 >= 90)
    source = tmp_path / "pairs.jsonl"
    write_rows(source, [
        {"source": target.replace("scrutinize", "scrutinise"), "target": target, "category": "spelling", "license": "CC0-1.0", "review_status": "human-reviewed"},
        {"source": "He have a pen.", "target": "He has a pen.", "category": "agreement", "license": "CC0-1.0", "review_status": "unreviewed"},
    ])
    manifest = prepare([source], tmp_path / "prepared", allow_weak_train=True, schema=2)
    assert manifest["evaluation"]["test"]["rows"] == 1
    assert manifest["splits"]["test"]["unsupported_tag_examples"] == 1
    assert manifest["splits"]["test"]["accepted"] == 0


def test_complete_multireference_scoring_and_failed_identity_output():
    rows = [{"source": "She have a book.", "references": ["She has a book.", "She owns a book."]},
            {"source": "Book.", "references": ["A carefully illustrated book."]},
            {"source": "We watched her duck.", "references": ["We watched her duck."]}]
    predictions = ["She owns a book.", "Book.", "We watched her duck."]
    report = score_predictions(rows, predictions, failures=1)
    assert report["sentences"] == 3
    assert report["true_positive_edits"] == 1
    assert report["missed_edits"] == 1
    assert report["clean_sentence_false_positive_rate"] == 0
    assert report["inference_failures"] == 1
    assert "not ERRANT" in report["scoring_note"]
    with pytest.raises(ValueError): score_predictions(rows, predictions[:-1])
    with pytest.raises(ValueError): score_predictions([{"source": "A book.", "references": []}], ["A book."])


def test_missing_clean_controls_and_unsafe_thresholds_trigger_explicit_no_edit():
    rows = [{"source": "She have a book.", "references": ["She has a book."]}]
    candidate = {**score_predictions(rows, ["She has a book."]), "threshold": .8}
    no_edit = {**score_predictions(rows, ["She have a book."]), "threshold": 1., "disable_model_edits": True}
    calibrated = choose_calibration([candidate], no_edit)
    assert candidate["clean_sentence_false_positive_rate"] is None
    assert calibrated["disable_model_edits"] is True
    assert calibrated["constraints_met"] is False
    assert calibrated["best_unconstrained"]["edit_f0_5"] == 1.
    candidate["clean_sentence_false_positive_rate"] = 0
    assert choose_calibration([candidate], no_edit, min_support=1)["constraints_met"] is True
    candidate["inference_failures"] = 1
    assert choose_calibration([candidate], no_edit)["constraints_met"] is False


def test_quantization_drift_cannot_enable_a_policy_that_fails_one_export():
    base = {"predicted_edits": 25, "edit_precision": .99, "clean_sentence_false_positive_rate": 0,
            "edit_f0_5": .5, "threshold": .8, "inference_failures": 0}
    no_edit = {**base, "threshold": 1., "predicted_edits": 0, "edit_f0_5": 0}
    assert choose_joint_calibration([base], [{**base, "edit_precision": .5}], no_edit)["disable_model_edits"] is True
    int8 = [base, {**base, "threshold": .95, "edit_f0_5": .4}]
    fp32 = [{**base, "edit_precision": .5}, {**base, "threshold": .95, "edit_f0_5": .3}]
    shared = choose_joint_calibration(int8, fp32, no_edit)
    assert shared["selected"]["threshold"] == .95
    assert shared["constraints_met"] is True


def test_deletion_and_article_guards_match_browser_contract():
    text = "Hello, world."
    proposal = decode_proposal(text, words(text), 1, "DELETE", .99, 2)
    assert apply_proposals(text, [proposal], .8) == "Hello world."
    text = "Hello,world."
    proposal = decode_proposal(text, words(text), 1, "DELETE", .99, 2)
    assert apply_proposals(text, [proposal], .8) == "Hello world."
    text = "We read books."
    assert decode_proposal(text, words(text), 2, "DELETE", .999, 2) is None
    text = "I have the book."
    assert decode_proposal(text, words(text), 2, "DELETE", .979, 2) is None
    assert decode_proposal(text, words(text), 2, "DELETE", .99, 2) is not None
    assert decode_proposal(text, words(text), 2, "DELETE", .99, 1) is None
    text = "She had had a book."
    assert decode_proposal(text, words(text), 1, "DELETE", .999, 2) is None
    text = "I have a university."
    assert decode_proposal(text, words(text), 2, "REPLACE_EXACT:an", .999, 2) is None
    text = "I have a hour."
    assert decode_proposal(text, words(text), 2, "REPLACE_EXACT:an", .999, 2) is not None


def test_overlapping_proposals_and_disabled_model_never_apply_twice():
    text = "hello world."
    proposed = [{"start": 0, "end": 5, "replacement": "Hello", "confidence": .99, "category": "case"},
                {"start": 0, "end": 0, "replacement": "Well, ", "confidence": .99, "category": "prepend"}]
    assert apply_proposals(text, proposed, .8, disabled=True) == text
    assert apply_proposals(text, proposed, .8) == "Well, hello world."
    assert apply_proposals(text, proposed[:1], .8, category_thresholds={"case": 1.}) == text
    assert apply_proposals(text, proposed[:1], 1., category_thresholds={"case": .8}) == text


def test_jfleg_import_preserves_every_reference_and_refuses_wrong_population(tmp_path):
    def fetch(name, revision):
        assert len(revision) == 40
        split, filename = name.split("/")
        count = 754 if split == "dev" else 747
        label = filename.rsplit(".", 1)[1]
        return "".join(f"{label} example {index}.\n" for index in range(count)).encode()
    manifest = materialize(tmp_path / "evaluation", fetch=fetch)
    assert manifest["training_allowed"] is False
    assert manifest["license"] == "CC-BY-NC-SA-4.0"
    row = json.loads((tmp_path / "evaluation/dev.jsonl").read_text().splitlines()[0])
    assert len(row["references"]) == 4
    assert row["review_status"] == "human-reference-benchmark"
    with pytest.raises(ValueError, match="already exists"): materialize(tmp_path / "evaluation", fetch=fetch)
    with pytest.raises(ValueError, match="population"):
        materialize(tmp_path / "short", fetch=lambda *_: b"One row.\n")


def test_experiment_selector_removes_conflicts_and_eval_only_data(tmp_path):
    source = tmp_path / "rows.jsonl"
    base = {"license": "CC0-1.0", "category": "mixed"}
    write_rows(source, [{**base, "source": "He have a book.", "target": "He has a book."},
                       {**base, "source": "He have a book.", "target": "He had a book."},
                       {**base, "source": "She have a pen.", "target": "She has a pen.", "evaluation_only": True},
                       {**base, "source": "They has a pen.", "target": "They have a pen."}])
    selected, counts = select_rows(source, 10, 10, 42, set(), set())
    assert len(selected) == 1
    assert selected[0]["source"] == "They has a pen."
    assert counts["conflicting_source"] == 1
    assert counts["invalid"] == 1


def test_cross_role_evaluation_strings_are_excluded_from_either_training_field(tmp_path):
    heldout = [{"source": "She have a pen.", "references": ["She has a pen.", "She owns a pen."]}]
    keys = evaluation_keys(heldout)
    path = tmp_path / "rows.jsonl"
    write_rows(path, [
        {"source": "She owns a pen.", "target": "She owns one pen.", "license": "CC0-1.0"},
        {"source": "She have pen.", "target": "She have a pen.", "license": "CC0-1.0"},
        {"source": "They has books.", "target": "They have books.", "license": "CC0-1.0"},
    ])
    selected, counts = select_rows(path, 10, 10, 42, keys, {group_id(value) for value in keys})
    assert counts["heldout_overlap"] == 2
    assert len(selected) == 1


@pytest.mark.parametrize("new_controls", [False, True])
def test_controlled_experiments_cap_equal_raw_counts_and_keep_eval_hashes(tmp_path, monkeypatch, new_controls):
    template, evaluation = tmp_path / "template", tmp_path / "evaluation"
    template.mkdir(); evaluation.mkdir()
    train = [{"source": f"He have a book for task {index}.", "target": f"He has a book for task {index}."} for index in range(8)]
    write_rows(template / "train.jsonl", train)
    for split in ["dev", "test"]:
        write_rows(template / f"{split}.jsonl", [{"source": f"She eat a {split} sandwich.", "target": f"She ate a {split} sandwich."}])
        write_rows(evaluation / f"{split}.jsonl", [{"source": f"She has a {split} pen.", "references": [f"She has a {split} pen."]}])
    weak = tmp_path / "weak.jsonl"
    write_rows(weak, [{"source": f"They has a pen for task {index}.", "target": f"They have a pen for task {index}.",
                      "license": "CC0-1.0", "category": "agreement", "review_status": "unreviewed"} for index in range(2)])
    args = Namespace(output=tmp_path / "run", rows=8, weak=[weak], max_scanned=20, epochs=2,
                     batch_size=2, evaluation_dir=evaluation, template=template, seed=42,
                     max_labels=100, learning_rate=.0005, local_files_only=True, device="cpu")
    if new_controls:
        excluded = tmp_path / "regression.jsonl"
        write_rows(excluded, [{"source": "I read the local checklist.", "target": train[0]["source"],
                               "references": ["I checked the local checklist."]}])
        args.exclude = [excluded]
        args.development_only = True
        args.keep_weight = 1.0
        args.shared_label_inventory = True
        args.arm = ["template-tiny64", "mixed-tiny64"]
    plan = build(args)
    assert plan["effective_raw_rows_per_arm"] == 4
    assert len(plan["arms"]) == (2 if new_controls else 4)
    assert {value["raw_train_rows"] for value in plan["dataset_manifests"].values()} == {4}
    for arm in plan["arms"]:
        assert arm["command"][arm["command"].index("--evaluation-dir") + 1] == str(evaluation.resolve())
        if new_controls:
            assert "--development-only" in arm["command"]
            assert arm["command"][arm["command"].index("--keep-weight") + 1] == "1.0"
    if new_controls:
        from pairs import hash_file
        inventories = [json.loads((args.output / "datasets" / name / "labels.json").read_text()) for name in ("template", "mixed")]
        assert inventories[0] == inventories[1]
        assert plan["shared_label_inventory"]["count"] == len(inventories[0])
        heldout_tags = set(edit_tags("She eat a sandwich.", "She ate a sandwich.", 2)[1]) - {"KEEP"}
        assert not heldout_tags.intersection(inventories[0])
        for name in ("template", "mixed"):
            raw = [json.loads(line) for line in (args.output / "datasets" / f"{name}-raw.jsonl").read_text().splitlines()]
            assert not any(row["source"] == train[0]["source"] for row in raw)
            manifest = json.loads((args.output / "datasets" / name / "manifest.json").read_text())
            assert manifest["labels_sha256"] == hash_file(args.output / "datasets" / name / "labels.json")
    from experiments import run
    started = []
    monkeypatch.setattr("experiments.subprocess.run", lambda command, **kwargs: started.append(command))
    if new_controls:
        # Deferred tagged test inputs cannot affect development-only runs.
        (args.output / "datasets/template/test.jsonl").write_text("unused changed tagged test\n")
        run(plan)
        assert len(started) == 2
        started.clear()
    changed_split = "dev" if new_controls else "test"
    (args.output / "datasets/template" / f"{changed_split}.jsonl").write_text("changed scored diagnostic\n")
    with pytest.raises(ValueError, match="Tagged diagnostic population changed"):
        run(plan)
    assert not started
    with pytest.raises(ValueError, match="already exists"): build(args)


def test_experiment_runner_rejects_code_drift_before_starting_training(tmp_path, monkeypatch):
    from experiments import run
    from pairs import hash_file
    source = tmp_path / "train.py"
    source.write_text("original frozen training code\n")
    plan = {"arms": [{"command": ["never-start-this"]}],
            "code_files": [{"path": str(source), "sha256": hash_file(source)}]}
    source.write_text("changed training code\n")
    monkeypatch.setattr("experiments.subprocess.run", lambda *args, **kwargs: pytest.fail("Training started after code drift"))
    with pytest.raises(ValueError, match="Training code changed"):
        run(plan)


def test_common_budget_equalizes_supported_rows_without_trimming_evaluation(tmp_path):
    manifests = {}
    for name, lengths in [("template", [3, 3, 3, 3]), ("mixed", [3, 7, 3])]:
        directory = tmp_path / name
        directory.mkdir()
        records = [{"pair_id": f"{name}-{index}", "tokens": ["word"] * length,
                    "source": "text", "target": "text"} for index, length in enumerate(lengths)]
        write_rows(directory / "train.jsonl", records)
        manifests[name] = {"splits": {"train": {"accepted": len(records)}}}
        (directory / "dev.jsonl").write_text("Retained full evaluation sentinel\n")
    result = common_training_population(tmp_path, manifests, lambda words: len(words) + 2, 6, 42)
    assert result["rows_per_arm"] == 2
    assert result["coverage"]["mixed"]["token_budget_rejected"] == 1
    for name in manifests:
        assert len((tmp_path / name / "train.jsonl").read_text().splitlines()) == 2
        assert (tmp_path / name / "dev.jsonl").read_text() == "Retained full evaluation sentinel\n"


def test_shared_full_source_verb_guards():
    fixture = json.loads(Path(__file__).with_name("edit-parity.json").read_text())
    for case in fixture["guarded"]:
        proposal = decode_proposal(case["source"], words(case["source"]), case["index"], case["tag"], .99, 2)
        actual = apply_proposals(case["source"], [proposal] if proposal else [], .8)
        assert actual == case["target"]


def test_case_only_conflicts_are_not_input_order_duplicates(tmp_path):
    source = tmp_path / "case.jsonl"
    write_rows(source, [
        {"source": "hello world.", "target": "Hello world.", "category": "mixed", "license": "CC0-1.0"},
        {"source": "hello world.", "target": "HELLO world.", "category": "mixed", "license": "CC0-1.0"},
        {"source": "She have a pen.", "target": "She has a pen.", "category": "mixed", "license": "CC0-1.0"}])
    selected, counts = select_rows(source, 10, 10, 42, set(), set())
    assert len(selected) == 1
    assert counts["conflicting_source"] == 1
    report = prepare([source], tmp_path / "prepared-case", allow_weak_train=True, schema=2)
    assert report["counts"]["conflicting_source"] == 1
    assert "hello world" not in (tmp_path / "prepared-case/train.jsonl").read_text().lower()


def test_nonfinite_keep_logits_fail_the_whole_sentence():
    np = pytest.importorskip("numpy")
    from evaluate import collect_proposals
    class Tokenizer:
        def encode(self, value, **kwargs): return [4]
        def __call__(self, batches, **kwargs):
            shape = (len(batches), len(batches[0]) + 2)
            class Encoded(dict):
                def word_ids(self, index): return [None, 0, 1, None]
            return Encoded({key: np.zeros(shape, dtype=np.int64) for key in
                            ["input_ids", "attention_mask", "token_type_ids"]})
    for invalid in [float("nan"), float("inf")]:
        def infer(feed):
            logits = np.zeros((*feed["input_ids"].shape, 2), dtype=np.float32)
            logits[0, 1, 0] = invalid
            return logits
        records, evidence = collect_proposals([{"source": "hello."}], Tokenizer(), infer, ["KEEP", "CASE:TITLE"], 64, 2)
        assert evidence["inference_failures"] == 1
        assert records == [{"proposals": [], "failed": True}]


def test_shared_protected_text_contract():
    from evaluate import protected_spans
    fixture = json.loads(Path(__file__).with_name("edit-parity.json").read_text())
    for case in fixture["protected"]:
        intervals = []
        for start, end in sorted(protected_spans(case["source"])):
            if intervals and start <= intervals[-1][1]: intervals[-1][1] = max(intervals[-1][1], end)
            else: intervals.append([start, end])
        assert [case["source"][start:end] for start, end in intervals] == case["fragments"]


def test_development_support_floor_cannot_be_faked_by_high_precision():
    from evaluate import qualified
    base = {"predicted_edits": 25, "edit_precision": 1., "clean_sentence_false_positive_rate": 0.,
            "edit_f0_5": .1, "threshold": .8, "inference_failures": 0}
    no_edit = {**base, "predicted_edits": 0, "edit_f0_5": 0., "threshold": 1.}
    for count in [0, 1, 5, 24]:
        candidate = {**base, "predicted_edits": count}
        report = choose_calibration([candidate], no_edit)
        assert report["disable_model_edits"] is True
        assert report["best_unconstrained"] == candidate
    assert choose_calibration([base], no_edit)["constraints_met"] is True
    assert not qualified(no_edit, .95, .02, min_support=1)
    for invalid in [{"edit_precision": .94}, {"clean_sentence_false_positive_rate": .03}, {"inference_failures": 1}]:
        assert not qualified({**base, **invalid}, .95, .02)
    assert choose_joint_calibration([base], [{**base, "predicted_edits": 24}], no_edit)["disable_model_edits"] is True
    assert choose_joint_calibration([base], [base], no_edit)["constraints_met"] is True
    assert choose_calibration([base], no_edit)["constraints"]["min_development_predicted_edits"] == 25
    with pytest.raises(ValueError, match="support"): choose_calibration([base], no_edit, min_support=0)
