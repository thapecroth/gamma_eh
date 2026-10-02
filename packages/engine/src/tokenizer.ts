/** BERT uncased BasicTokenizer + greedy WordPiece, retaining original UTF-16 spans. */
export interface WordToken { text: string; start: number; end: number }
export interface EncodedChunk { ids: number[]; positions: Array<{word: WordToken; position: number}> }

function isChinese(character: string): boolean {
  const point = character.codePointAt(0)!;
  return (point >= 0x4e00 && point <= 0x9fff) || (point >= 0x3400 && point <= 0x4dbf) ||
    (point >= 0x20000 && point <= 0x2a6df) || (point >= 0x2a700 && point <= 0x2b73f) ||
    (point >= 0x2b740 && point <= 0x2b81f) || (point >= 0x2b820 && point <= 0x2ceaf) ||
    (point >= 0xf900 && point <= 0xfaff) || (point >= 0x2f800 && point <= 0x2fa1f);
}

export function splitWords(text: string): WordToken[] {
  return [...text.matchAll(/[\p{L}]+(?:['’][\p{L}]+)*|[0-9]+|[^\p{L}\p{N}_\s]/gu)]
    .map(match => ({text: match[0], start: match.index, end: match.index + match[0].length}));
}

export class WordPieceTokenizer {
  private vocab: Map<string, number>;
  constructor(vocabulary: string) {
    this.vocab = new Map(vocabulary.trimEnd().split(/\r?\n/u).map((token, i) => [token, i]));
    for (const token of ['[CLS]', '[SEP]', '[UNK]']) {
      if (!this.vocab.has(token)) throw new Error(`Tokenizer is missing ${token}`);
    }
  }

  private id(token: string): number { return this.vocab.get(token) ?? this.vocab.get('[UNK]')!; }

  encodeWord(word: string): number[] {
    // Training uses BertTokenizerFast(is_split_into_words=True). BasicTokenizer
    // splits punctuation within contractions and strips accents after lowercasing.
    const normalized = [...word.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu, '')]
      .map(character => isChinese(character) ? ' ' + character + ' ' : character).join('');
    const basic = normalized.match(/[\p{L}\p{N}]+|[^\p{L}\p{N}\s]/gu) ?? [];
    const result: number[] = [];
    for (const token of basic) {
      const characters = [...token];
      if (characters.length > 100) { result.push(this.id('[UNK]')); continue; }
      let start = 0;
      const pieces: number[] = [];
      while (start < characters.length) {
        let end = characters.length;
        let found: number | undefined;
        while (end > start) {
          const piece = (start > 0 ? '##' : '') + characters.slice(start, end).join('');
          found = this.vocab.get(piece);
          if (found !== undefined) break;
          end--;
        }
        if (found === undefined) { pieces.length = 0; pieces.push(this.id('[UNK]')); break; }
        pieces.push(found);
        start = end;
      }
      result.push(...pieces);
    }
    return result.length ? result : [this.id('[UNK]')];
  }

  chunks(text: string, maxLength: number): EncodedChunk[] {
    if (!Number.isInteger(maxLength) || maxLength < 4 || maxLength > 512) throw new Error('Invalid model context length');
    const words = splitWords(text);
    const chunks: EncodedChunk[] = [];
    let current: EncodedChunk = {ids: [this.id('[CLS]')], positions: []};
    const flush = () => {
      if (!current.positions.length) return;
      current.ids.push(this.id('[SEP]'));
      chunks.push(current);
      current = {ids: [this.id('[CLS]')], positions: []};
    };
    for (const word of words) {
      const pieces = this.encodeWord(word.text);
      if (pieces.length > maxLength - 2) { flush(); continue; }
      if (current.ids.length + pieces.length >= maxLength) flush();
      current.positions.push({word, position: current.ids.length});
      current.ids.push(...pieces);
      // Sentence boundaries improve context isolation for the short-text baseline.
      if (/[.!?]/u.test(word.text) && word.text.length === 1) flush();
    }
    flush();
    return chunks;
  }
}
