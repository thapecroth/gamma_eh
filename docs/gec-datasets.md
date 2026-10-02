# Natural correction datasets and release rights

Research verified on 2026-10-01. The recommendation is to measure the current
model on independently annotated natural writing, then choose training sources
whose terms fit an unrestricted public model release. Synthetic template scores
cannot establish that release's real-world quality.

## Benchmark shortlist

| Dataset | Population and purpose | Availability and terms |
| --- | --- | --- |
| [JFLEG](https://github.com/keisks/jfleg) | 754 development / 747 test sentences, four human fluency corrections each. Measures learner-writing correction, including rewrites. | Public download, CC BY-NC-SA 4.0. Use official corpus GLEU. |
| [CWEB](https://github.com/SimonHFL/CWEB) | 6,729 development / 6,845 test sentences, two annotators. Low error density web text tests unnecessary corrections. | Public original/tokenized text and M2 annotations, CC BY-NC-SA 4.0. Use standard ERRANT. |
| [BEA-2019 W&I+LOCNESS](https://www.cl.cam.ac.uk/research/nl/bea2019st/) | 4,384 development / 4,477 test sentences spanning learner proficiency and native essays. | Development gold is downloadable; official test gold is withheld. Noncommercial research/education terms and LOCNESS redistribution restrictions. |
| [CoNLL-2014](https://www.comp.nus.edu.sg/~nlp/conll14st.html) | 1,312 test sentences from 50 learner essays; historical comparison. | Public test archive; inspect its accompanying terms. Use official M2 scoring, distinguish original annotations from expanded alternatives. |
| [ErAConD](https://github.com/yuanxun-yx/eracond) | 186 learner-chat dialogs, 2,454 sentences, focused on movies and COVID. Supplementary conversational coverage. | Data-only repository published under MIT, with no separate restrictive corpus notice. No canonical test split; a fixed-model full-corpus audit must be labeled separately from the paper's protocol. |
| [GMEG](https://github.com/grammarly/GMEG) | About 6,000 sentences across learner essays, Wikipedia and Yahoo Answers, with four human corrections. | Source-dependent access; Yahoo requires prior corpus approval and a separate request. No explicit blanket corpus license found. Defer automated import. |

We ran JFLEG and CWEB in the [public benchmark](public-benchmark.md). They cover
different failure modes; neither proves reliability across email, chat,
technical prose, dialects, accessibility tools, or every browser.

The BEA organizers now direct test evaluation to
[Codabench](https://www.codabench.org/competitions/10960/) using ERRANT 3.0.0.
Original 2019 results used ERRANT 2.0.0. Always state the scorer version, split,
track, and tokenization; a local development score is not an official test result.

[ErAConD's paper](https://aclanthology.org/2022.naacl-main.5/) uses severity-3
edits and modified ERRANT. Standard all-edit ERRANT on the entire corpus would
be a new audit, rather than a reproduction of those published scores. Its small
topic range also limits generalization.

## Training shortlist

| Dataset | Type and size | Decision for a public model |
| --- | --- | --- |
| [W&I training](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data) | 34,308 professionally annotated learner sentences. No native LOCNESS training split. | Strong research fine-tuning source; noncommercial restrictions require separate review before unrestricted weight release. |
| [FCE](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data) | 28,350 training sentences from real exam writing. | Human supervision, with the same noncommercial caution. Preserve separate dev/test splits. |
| [NUCLE](https://www.comp.nus.edu.sg/~nlp/corpora.html) | 57,151 sentences from approximately 1,400 essays, corrected by instructors. | Access requires an agreement/request. [License](https://www.comp.nus.edu.sg/~nlp/conll14st/nucle_license.pdf) restricts use to noncommercial research/trials. |
| [Lang-8 / NAIST](https://sites.google.com/site/naistlang8corpora) | BEA standardized English subset has 1,037,561 sentences with crowd corrections. | Research/education access; commercial permission requires the owner. Filter inconsistent, incomplete and meaning-changing corrections. |
| [cLang-8](https://github.com/google-research-datasets/clang8) | 2,372,119 English pairs: natural learner inputs with gT5-generated corrections. | CC BY-NC-SA 4.0 data; Apache code does not remove data restrictions. These targets are model supervision, not human gold. |
| [C4_200M](https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction) | 200 million synthetic corruptions of C4 sentences. | Useful pretraining candidate. The official CC BY 4.0 grant covers corruption edits; underlying web-text provenance and terms still matter. |
| [WikEd / wikiedits](https://github.com/snukky/wikiedits) | Wikipedia revisions; the [authors](https://emjotde.github.io/publications/pdf/mjd.poltal2014.draft.pdf) report 12.13 million sentences and about 14 million edits. | Corpus inherits CC BY-SA 3.0 per the paper. Preserve attribution/share-alike terms and filter content changes before using edits as grammar labels. |
| [GitHub Typo Corpus](https://github.com/mhagiwara/github-typo-corpus) | Approximately 353,000 edits across more than 15 languages; many spelling/mechanical edits in software prose. | Each snippet retains its source repository's license. Select English, retain repository/commit/license provenance, and filter incomplete text and non-grammar changes. |

For publishable training, prioritize consented human corrections, a reviewed
WikEd subset, and a source-license-filtered English GitHub Typo subset. This is
an engineering recommendation, not a finding that every record or derived
weight has unrestricted release rights. Downloadability, a hosting site's
license tag, code licensing, corpus licensing, and weight-release permission
are separate questions.

Learner corpus split counts follow the
[BEA-2019 shared-task report](https://aclanthology.org/W19-4406/).

Do not merge JFLEG/CWEB gold, teacher rewrites of their sentences, or their
sentence-level predictions into training. Keep weak teacher labels and C4
corruptions train-only, with separate provenance. Split natural training by
document/author and correction group before creating variants, deduplicate
against reserved benchmarks, and retain an untouched future test population.

## What must improve before a reliability claim

1. Collect licensed, independently reviewed development corrections and clean
   controls in the actual writing domains. Include names, technical words,
   contractions, dialect variation, and real-word spelling errors.
2. Measure how many full examples the edit representation supports before
   scaling. The earlier bounded C4 study supported 37.8% with schema 1 and
   59.6% with schema 2; those are sample coverage measurements, not accuracy.
3. Compare compact students on identical data. Calibrate precision, clean-text
   change rate, and sufficient edit support on development only. Preserve the
   existing disabled-policy fallback when no candidate qualifies.
4. Freeze code, weights and policy, then report standard benchmark scores,
   no-change baselines, error counts, clean-text changes, and browser failures.
   Improvements must survive new natural test data and human review.

The [completed model quality study](model-quality-results.md) already shows
that adding weak data and increasing model size does not by itself qualify a
release. The current benchmark provides a reproducible baseline for the next
round; it does not convert the experimental classifier into a general writing
model.
