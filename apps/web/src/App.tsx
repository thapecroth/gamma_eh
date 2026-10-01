import { useEffect, useRef, useState } from 'react';
import { applySuggestion, applySuggestions, type AnalysisResult, type Suggestion } from '@gamma/engine';

const EXAMPLES = [
  {
    id: 'email', label: 'An email', title: 'A little follow-up',
    description: 'A few familiar typos in an everyday email.',
    text: 'Hi Alex,\n\nI recieved your message about the project. We has a few ideas to share, and I would definately love to hear what you think.\n\nMy freind is putting together a seperate document with the details. Are you avaliable tomorrow?\n\nThanks for your help!',
  },
  {
    id: 'everyday', label: 'Everyday writing', title: 'Notes from the week',
    description: 'Try subject–verb agreement and a repeated word.',
    text: 'She have a new notebook. They is excited about the project.\n\nI have a few ideas for the the next meeting. My friend walk to the library every morning.\n\nA little clarity goes a long way.',
  },
  {
    id: 'spelling', label: 'A spelling check', title: 'Words in progress',
    description: 'Spelling suggestions from the bundled local dictionary.',
    text: 'Speling matters in a sentnce.\n\nI beleive a small change can make a big diffrence. It is neccessary to check the details before writting a message.\n\nYour words. Your voice.',
  },
  {
    id: 'ai', label: 'Try local AI', title: 'A tiny model at work',
    description: 'This example enables the experimental model on your device.',
    text: 'The students has a notebook.\n\nMy teacher go to school every morning.',
  },
] as const;

type WorkerResponse = { requestId: number; result?: AnalysisResult; error?: string };

function Mark({ compact = false }: { compact?: boolean }) {
  return <a className="brand" href="#" aria-label={compact ? 'Gamma EH home' : 'Gamma EH playground home'}>
    <img className="brand-icon" src="/favicon.svg" width="32" height="35" alt="" />
    <span>gamma<span className="brand-suffix"> eh</span></span>
    {!compact && <span className="brand-tag">PLAYGROUND</span>}
  </a>;
}

function Arrow() {
  return <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
    <path d="M5 12h14m-5-5 5 5-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
  </svg>;
}

function SuggestionCard({ suggestion, onAccept, onDismiss }: { suggestion: Suggestion; onAccept: () => void; onDismiss: () => void }) {
  return <article className="suggestion-card">
    <div className="suggestion-label"><span className={`category-dot ${suggestion.category}`} aria-hidden="true" />{suggestion.category}<span className="suggestion-origin">{suggestion.source === 'model' ? 'LOCAL MODEL' : suggestion.source.toUpperCase()}</span></div>
    <p className="correction"><span className="original">{suggestion.original || '(insert)'}</span><Arrow /><span className="replacement">{suggestion.replacement || '(remove)'}</span></p>
    <p className="suggestion-message">{suggestion.message}</p>
    <div className="suggestion-actions"><button className="accept-button" onClick={onAccept}>Accept <span aria-hidden="true">✓</span></button><button className="dismiss-button" onClick={onDismiss} aria-label={`Dismiss: ${suggestion.message}`}>Dismiss</button></div>
  </article>;
}

export default function App() {
  const [exampleId, setExampleId] = useState<string>(EXAMPLES[0].id);
  const [text, setText] = useState<string>(EXAMPLES[0].text);
  const [useAI, setUseAI] = useState(false);
  const [revision, setRevision] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [status, setStatus] = useState<'checking' | 'ready' | 'error'>('checking');
  const [error, setError] = useState('');
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [notice, setNotice] = useState('');
  const [undoText, setUndoText] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const requestId = useRef(0);
  const textRef = useRef(text);
  const editorRef = useRef<HTMLTextAreaElement>(null);
  textRef.current = text;

  useEffect(() => {
    const worker = new Worker(new URL('./inference-worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.requestId !== requestId.current) return;
      if (message.result && message.result.text === textRef.current) {
        setResult(message.result);
        setStatus('ready');
        setError('');
      } else if (message.error) {
        setStatus('error');
        setError('The local checker could not check this draft. Try reloading this page.');
      }
    };
    worker.onerror = () => { setStatus('error'); setError('The local checker could not start. Try reloading this page.'); };
    return () => { worker.terminate(); workerRef.current = null; };
  }, []);

  useEffect(() => {
    const id = ++requestId.current;
    const timer = window.setTimeout(() => workerRef.current?.postMessage({ requestId: id, text, useAI }), 450);
    return () => window.clearTimeout(timer);
  }, [text, useAI, revision]);

  const example = EXAMPLES.find((item) => item.id === exampleId) ?? EXAMPLES[0];
  const current = result?.text === text && status === 'ready';
  const suggestions = current ? result.suggestions.filter((suggestion) => !dismissed.has(suggestion.id)) : [];
  const words = text.trim() ? text.trim().split(/\s+/u).length : 0;
  const backendLabel = result?.backend === 'webgpu' ? 'Local AI · WebGPU' : result?.backend === 'wasm' ? 'Local AI · CPU' : 'Local rules';

  function invalidateCheck() {
    // Invalidate immediately, including resets to the same text and mode changes.
    ++requestId.current;
    setResult(null);
    setStatus('checking');
    setDismissed(new Set());
    setRevision((previous) => previous + 1);
  }

  function updateDraft(next: string, message = '') {
    textRef.current = next;
    setText(next);
    setUndoText(null);
    setNotice(message);
    invalidateCheck();
  }

  function loadExample(item: typeof EXAMPLES[number]) {
    setExampleId(item.id);
    if (item.id === 'ai') setUseAI(true);
    updateDraft(item.text, `${item.label} loaded.`);
  }

  function accept(suggestion: Suggestion) {
    if (!current || textRef.current !== result.text) return;
    try {
      const previous = textRef.current;
      updateDraft(applySuggestion(previous, suggestion), 'Suggestion accepted.');
      setUndoText(previous);
      editorRef.current?.focus();
    } catch { setNotice('Your text changed. Suggestions will refresh.'); }
  }

  function acceptAll() {
    if (!current || !suggestions.length || textRef.current !== result.text) return;
    try {
      const previous = textRef.current;
      updateDraft(applySuggestions(previous, suggestions), 'Suggestions accepted.');
      setUndoText(previous);
      editorRef.current?.focus();
    } catch { setNotice('Your text changed. Suggestions will refresh.'); }
  }

  async function copyDraft() {
    try {
      await navigator.clipboard.writeText(textRef.current);
      setNotice('Draft copied to clipboard.');
    } catch {
      setNotice('Couldn’t copy automatically. Select your draft and copy it instead.');
      editorRef.current?.focus();
      editorRef.current?.select();
    }
  }

  return <div className="app-shell">
    <header className="site-header"><Mark /><nav aria-label="Main navigation"><a className="active-nav" href="#playground">Playground</a><a href="#how-it-works">How it works</a><a href="https://github.com/thapecroth/gamma_eh" target="_blank" rel="noreferrer">Source code <span aria-hidden="true">↗</span></a></nav></header>
    <main>
      <section className="intro" aria-labelledby="intro-title">
        <div className="eyebrow"><span /> A SMALL SPACE FOR BETTER WRITING</div>
        <h1 id="intro-title">A little help.<br /><em>All on your device.</em></h1>
        <p>Meet your local writing assistant. Try a draft, explore a suggestion,<br className="desktop-break" /> and keep what sounds like you. No extension needed.</p>
        <div className="intro-note"><span aria-hidden="true">↙</span> Go ahead. Make it your own.</div>
        <div className="intro-badges"><span><span aria-hidden="true">✓</span> No installation</span><span><span aria-hidden="true">✓</span> No account</span><span><span aria-hidden="true">✓</span> No text uploads</span></div>
      </section>
      <section id="playground" className="playground" aria-label="Writing playground">
        <div className="example-picker"><span className="example-label" id="example-label">Start with an example</span><div className="example-buttons" role="group" aria-labelledby="example-label">
          {EXAMPLES.map((item) => <button key={item.id} aria-pressed={exampleId === item.id} onClick={() => loadExample(item)}>{item.label}{item.id === 'ai' && <span className="ai-badge" aria-hidden="true">✧</span>}</button>)}
        </div></div>
        <div className="workspace">
          <div className="editor-pane">
            <div className="pane-toolbar"><span className="document-title"><span aria-hidden="true">▤</span>{example.title}</span><span className="language">English</span></div>
            <div className="draft-actions" aria-label="Draft actions">
              <button onClick={() => updateDraft(example.text, 'Example reset.')}>Reset example</button>
              <button disabled={!text} onClick={() => { updateDraft('', 'Draft cleared.'); editorRef.current?.focus(); }}>Clear</button>
              <button disabled={undoText === null} onClick={() => { if (undoText !== null) updateDraft(undoText, 'Last correction undone.'); }}>Undo correction</button>
              <button className="copy-button" disabled={!text} onClick={() => void copyDraft()}>Copy text <span aria-hidden="true">↗</span></button>
            </div>
            <label className="sr-only" htmlFor="writing-editor">Your writing</label>
            <textarea id="writing-editor" ref={editorRef} className="writing-editor" spellCheck={false} maxLength={20_000} value={text} onChange={(event) => updateDraft(event.target.value)} placeholder="A blank page, a fresh start. Type or paste your own writing…" aria-describedby="privacy-note example-description" />
            <div className="editor-footer"><span>{words} {words === 1 ? 'word' : 'words'}<span className="footer-divider">·</span>{text.length.toLocaleString()} / 20,000 characters</span><span className="saved-indicator"><span /> In this tab only</span></div>
          </div>
          <aside className="review-pane" aria-label="Writing suggestions">
            <div className="review-heading"><div><h2>A second pair of eyes <span aria-hidden="true">✧</span></h2><p>You choose what to change.</p></div><span className="issue-count" aria-label={`${suggestions.length} suggestions`}>{suggestions.length}</span></div>
            <div className="checker-settings">
              <label className="toggle-label"><input type="checkbox" checked={useAI} onChange={(event) => { setUseAI(event.target.checked); invalidateCheck(); }} /><span className="switch" aria-hidden="true" />Local AI<span className="experimental">EXPERIMENTAL</span></label>
              <div className="backend-status" role="status"><span className={`status-dot ${status}`} />{status === 'checking' ? useAI ? 'Checking on your device…' : 'Checking local rules…' : status === 'error' ? 'Checker unavailable' : backendLabel}{current && <span className="check-time">{Math.round(result.elapsedMs).toLocaleString()} ms</span>}</div>
            </div>
            {result?.modelError && current && <p className="model-warning">Local AI couldn’t load. Spelling and rule suggestions still work. Try switching Local AI off and on again.</p>}
            {useAI && <p className="model-note">A tiny model trained on synthetic examples. Experimental suggestions may miss real-world errors.</p>}
            <div className="suggestion-list" aria-live="polite" aria-busy={status === 'checking'}>
              {status === 'checking' ? <div className="empty-state"><span className="checking-symbol" aria-hidden="true">✧</span><p>Giving your words a look.</p><span>{useAI ? 'The first local AI check may take a moment.' : 'Spelling and grammar checks run right here.'}</span></div>
                : status === 'error' ? <div className="empty-state"><p>We couldn’t check this draft.</p><span>{error}</span></div>
                : suggestions.length ? suggestions.map((suggestion) => <SuggestionCard key={suggestion.id} suggestion={suggestion} onAccept={() => accept(suggestion)} onDismiss={() => { setDismissed((previous) => new Set([...previous, suggestion.id])); setNotice('Suggestion dismissed.'); }} />)
                : <div className="empty-state"><span className="all-clear-symbol" aria-hidden="true">✓</span><p>{text.trim() ? 'No suggestions from this checker.' : 'Make room for your words.'}</p><span>{text.trim() ? 'Keep writing in your own voice.' : 'Type something or choose an example to get started.'}</span></div>}
            </div>
            <div className="review-footer"><button className="accept-all-button" disabled={!suggestions.length || !current} onClick={acceptAll}>Accept all suggestions <Arrow /></button></div>
          </aside>
        </div>
        <div className="playground-caption"><p id="example-description">{example.description}</p><p className="action-notice" role="status">{notice}</p></div>
      </section>
      <div className="beneath-editor"><p id="privacy-note"><span aria-hidden="true">◇</span>Checks run in this tab. We don’t upload or store your draft.</p><span>Your words. Your call.</span></div>
      <section id="how-it-works" className="how-it-works" aria-labelledby="how-title">
        <div className="how-heading"><span className="eyebrow">A LITTLE HELP, WITHOUT THE SETUP</span><h2 id="how-title">Open a tab. Find your flow.</h2></div>
        <div className="principles">
          <div><span className="step-number">01</span><h3>Bring your words</h3><p>Start with an example or paste your own draft. No extension, sign-up, or API key.</p></div>
          <div><span className="step-number">02</span><h3>Check on your device</h3><p>Spelling and rules work instantly. Turn on Local AI to try the bundled experimental model.</p></div>
          <div><span className="step-number">03</span><h3>Keep your voice</h3><p>Accept a change, dismiss it, or undo your last correction. Copy your text when you’re ready.</p></div>
        </div>
      </section>
    </main>
    <footer className="site-footer"><Mark compact /><p>Small model. Open code. Your words.</p><span>Apache 2.0 · Open source</span></footer>
  </div>;
}
