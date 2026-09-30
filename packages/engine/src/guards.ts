import { canonicalSpelling } from './rules';

// The v1 model knows a narrow article vocabulary. Reject neural article edits
// outside known sound classes; first-letter heuristics fail for hour/university.
const articleHeads: Record<string, 'a' | 'an'> = {
  apple: 'an', orange: 'an', egg: 'an', umbrella: 'an', envelope: 'an', idea: 'an',
  hour: 'an', honest: 'an', honor: 'an', heir: 'an',
  book: 'a', pencil: 'a', bicycle: 'a', camera: 'a', notebook: 'a', ticket: 'a',
  friend: 'a', message: 'a', pen: 'a', project: 'a', library: 'a',
  university: 'a', user: 'a', unicorn: 'a', european: 'a', one: 'a',
};

export function validArticleEdit(article: string, nextWord: string): boolean {
  return (article === 'a' || article === 'an') && articleHeads[canonicalSpelling(nextWord)] === article;
}
