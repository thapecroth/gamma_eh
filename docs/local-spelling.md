# Local dictionary spelling

The web editor and Chrome extension share a deterministic spelling fallback in
`packages/engine/src/spelling.ts`. It works with Local AI disabled and when the
model cannot load. Typed text stays in the existing local inference workers;
the frequency data is bundled into their JavaScript. There is no runtime
dictionary download or new dependency.

## Retrieval and ranking

The bundled dictionary accepts 82,769 lowercase ASCII English words from the
pinned [SymSpell frequency dictionary](https://github.com/wolfgarbe/SymSpell#frequency-dictionary).
Accepted words are checked with a set lookup and never corrected by fuzzy
matching. The existing curated typo rules retain precedence.

A lazy symmetric-delete index covers the 20,000 most frequent correction
candidates. Indexing only these candidates reduces startup and memory cost and
avoids offering rare words, while the larger accepted set protects rare valid
words. A seven-character prefix and at most two deletions create at most 29
lookup keys per word; lookup retrieves candidates rather than scanning the
dictionary. Banded optimal-string-alignment distance verifies full-word edits,
including adjacent transpositions. Tokens shorter than five characters allow
only one edit; longer tokens allow two. Tokens over 32 characters are skipped.

Rank by edit distance, then prefer a single missing doubled letter, then word
frequency with deterministic lexical ties. The doubled-letter preference keeps
`helo` closer to `hello` than to frequent alternatives such as `help`. Abstain
when the best and second-best candidates with the same distance and doubled-letter
preference have similar frequency: require a 2x margin for one edit and 4x for two. These are conservative
heuristics, not calibrated probabilities or measured real-world accuracy.
Results, including misses, are cached by lowercase word in a bounded 2,048-entry
FIFO cache per worker. No text or cache is persisted.
Words already covered by curated rules are skipped before fuzzy lookup, so
those corrections do not initialize the candidate index.

## Real words and safety

`halo` is a valid noun, so fuzzy dictionary matching leaves it alone. An explicit
standalone greeting heuristic offers `hello` when the entire field contains
`halo`, optional whitespace and sentence punctuation. Its message states that
both are valid and its confidence is tentative. “A halo around the moon” and
“Halo is a game” stay unchanged. Suggestions require the usual user acceptance;
source offsets remain UTF-16 and stale edits are rejected.

Consume whole Unicode tokens before choosing plain ASCII words. Skip mixed-case
identifiers (including Unicode connector punctuation and joiners), numbers,
contractions, hyphenated compounds (including Unicode and nonbreaking hyphens),
acronyms, capitalized names inside sentences, URLs, emails, paths, code and hashtags. A small accepted
list protects current technical names absent from the older corpus. This is
English-only and cannot recognize every name or specialist word. The nearest
dictionary word is not always the intended word. Multiword splits/joins and
general real-word error correction remain outside this fallback.
Backtick runs protect multiline code and embedded shorter backticks. An
unfinished delimiter protects the remaining draft until it is closed.

## Provenance and validation

Run `node scripts/generate-spelling-dictionary.mjs` to regenerate from the
pinned, checksum-verified download, or pass a local source filename for offline
regeneration. Generation is opt-in and separate from normal builds.
`packages/engine/dictionary-provenance.json` records the source revision,
checksums and transformation. See [dictionary notices](../licenses/spelling/README.md)
for Google Books Ngrams, SCOWL and SymSpell attribution and licenses. Release
archives include these notices separately from the app's source license.

`tests/spelling.test.ts` covers unseen typos, valid words, abstention, greetings,
case, Unicode offsets, protected content, rule precedence, stale edits and
bounded distance against an independent full-matrix reference. Browser smoke
checks cover dictionary suggestions and acceptance with AI disabled in both
shipping workers. These are regression fixtures, not a grammar-accuracy score.

Run `node --expose-gc scripts/benchmark-spelling.mjs` for a reproducible Node/V8
microbenchmark: dictionary and index startup, retained heap growth, 1,000
deterministic typo queries, cached queries and a 20,000-character draft. It writes
`artifacts/spelling-benchmark.json`. This measures synthetic fixtures on the
current machine and does not establish mobile, browser, or physical-GPU speed.
