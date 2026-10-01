# English spelling dictionary attribution

Gamma EH bundles a transformed version of Wolf Garbe's **SymSpell English
frequency dictionary**, `frequency_dictionary_en_82_765.txt`, at revision
`c239062ae02961df18ab7da1671d01b4388204e0`:
https://github.com/wolfgarbe/SymSpell/blob/c239062ae02961df18ab7da1671d01b4388204e0/SymSpell/frequency_dictionary_en_82_765.txt

The dictionary intersects **Google Books Ngrams** (Google, distributed under
[Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/))
with **SCOWL — Spell Checking Oriented Word Lists** (Kevin Atkinson and the
contributors credited in SCOWL-Copyright). SymSpell's
[source description](https://github.com/wolfgarbe/SymSpell#frequency-dictionary)
identifies both sources and their licenses. These data notices apply separately
from Gamma EH's Apache-2.0 source license. No upstream endorsement is implied.

Changes: retain lowercase ASCII words, sort by frequency and lexical ties,
and keep counts for the first 20,000 correction candidates. All 82,769 accepted
words remain available to protect valid vocabulary. The transformation is
reproducible with `scripts/generate-spelling-dictionary.mjs`; the source and
generated SHA-256 checksums are in `packages/engine/dictionary-provenance.json`.

SymSpell-LICENSE retains Wolf Garbe's upstream MIT notice. SCOWL-Copyright is
an unchanged copy from `en-wl/wordlist` revision
`b22230cc5250887737fdefe9ca4c9d9d01230eaa`, `scowl/Copyright` (the SCOWLv1
source referenced by SymSpell). Both files are included in Chrome and web release
archives. The TypeScript algorithm implementation is original Gamma EH code;
it uses the published symmetric-delete idea and bounded OSA verification.
