import { useEffect, useRef, useState } from 'react';
import { applySuggestion, applySuggestions, type AnalysisResult, type Suggestion } from '@gamma/engine';

const SAMPLE = "A little clarity goes a long way.\n\nShe go to the library every morning. I recieved a message about the new project, and we has a lot of ideas. There is something special about putting your thoughts into words.\n\nWrite freely. We'll help with the small things.";

type WorkerResponse = { requestId: number; result?: AnalysisResult; error?: string };

function Mark({ compact = false }: { compact?: boolean }) {
  return <a className="brand" href="#" aria-label="Gamma EH home"><span className="brand-icon" aria-hidden="true">γ</span><span>gamma<span className="brand-suffix">eh</span></span>{!compact && <span className="brand-tag">OPEN SOURCE</span>}</a>;
}

function Arrow() {
  return <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true"><path d="M5 12h14m-5-5 5 5-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

function SuggestionCard({ suggestion, onAccept, onDismiss }: { suggestion: Suggestion; onAccept: () => void; onDismiss: () => void }) {
  return <article className="suggestion-card">
    <div className="suggestion-label"><span className={`category-dot ${suggestion.category}`} />{suggestion.category}<span className="suggestion-origin">{suggestion.source === 'model' ? 'LOCAL MODEL' : suggestion.source.toUpperCase()}</span></div>
    <p className="correction"><span className="original">{suggestion.original || '(insert)'}</span><Arrow /><span className="replacement">{suggestion.replacement || '(remove)'}</span></p>
    <p className="suggestion-message">{suggestion.message}</p>
    <div className="suggestion-actions"><button className="accept-button" onClick={onAccept}>Accept <span aria-hidden="true">↵</span></button><button className="dismiss-button" onClick={onDismiss} aria-label={`Dismiss: ${suggestion.message}`}>Dismiss</button></div>
  </article>;
}

export default function App() {
  const [text, setText] = useState(SAMPLE);
  const [useAI, setUseAI] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [status, setStatus] = useState<'checking' | 'ready' | 'error'>('checking');
  const [error, setError] = useState('');
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [notice, setNotice] = useState('');
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
        setError(message.error);
      }
    };
    worker.onerror = () => { setStatus('error'); setError('The local checker could not start. Try reloading this page.'); };
    return () => { worker.terminate(); workerRef.current = null; };
  }, []);

  useEffect(() => {
    const id = ++requestId.current;
    setStatus('checking');
    setDismissed(new Set());
    const timer = window.setTimeout(() => workerRef.current?.postMessage({ requestId: id, text, useAI }), 450);
    return () => window.clearTimeout(timer);
  }, [text, useAI]);

  const current = result?.text === text && status === 'ready';
  const suggestions = current ? result.suggestions.filter((suggestion) => !dismissed.has(suggestion.id)) : [];
  const words = text.trim() ? text.trim().split(/\s+/u).length : 0;
  const backendLabel = result?.backend === 'webgpu' ? 'Local AI · WebGPU' : result?.backend === 'wasm' ? 'Local AI · CPU' : 'Local rules';

  function accept(suggestion: Suggestion) {
    if (!current || textRef.current !== result.text) return;
    try {
      const next = applySuggestion(textRef.current, suggestion);
      textRef.current = next;
      setText(next);
      setNotice('Suggestion accepted.');
      editorRef.current?.focus();
    } catch {
      setNotice('Your text changed. Suggestions will refresh.');
    }
  }

  function acceptAll() {
    if (!current || !suggestions.length || textRef.current !== result.text) return;
    try {
      const next = applySuggestions(textRef.current, suggestions);
      textRef.current = next;
      setText(next);
      setNotice('Suggestions accepted.');
      editorRef.current?.focus();
    } catch {
      setNotice('Your text changed. Suggestions will refresh.');
    }
  }

  return <div className="app-shell">
    <header className="site-header"><Mark /><nav aria-label="Main navigation"><a href="https://github.com/A-IDEAL/gamma_eh" target="_blank" rel="noreferrer">Source code <span aria-hidden="true">↗</span></a><a className="extension-link" href="https://github.com/A-IDEAL/gamma_eh#chrome-extension" target="_blank" rel="noreferrer">Get the extension <Arrow /></a></nav></header>
    <main>
      <section className="intro"><div className="eyebrow"><span /> YOUR WORDS. A LITTLE CLEARER.</div><h1>Good writing starts<br />with <em>your voice.</em></h1><p>A second pair of eyes for the little things.<br />Private by design. Open to everyone.</p><div className="intro-note"><span aria-hidden="true">↙</span> Give it a try. Your words stay here.</div></section>
      <section className="workspace" aria-label="Writing assistant">
        <div className="editor-pane"><div className="pane-toolbar"><span className="document-title"><svg width="17" height="19" viewBox="0 0 17 19" fill="none" aria-hidden="true"><path d="M3 1h7l4 4v13H3V1Z" stroke="currentColor" strokeWidth="1.3" /><path d="M10 1v4h4M6 9h5M6 12h5" stroke="currentColor" strokeWidth="1.3" /></svg>Untitled draft</span><span className="language">English <span aria-hidden="true">⌄</span></span></div><label className="sr-only" htmlFor="writing-editor">Your writing</label><textarea id="writing-editor" ref={editorRef} className="writing-editor" spellCheck={false} maxLength={20_000} value={text} onChange={(event) => { textRef.current = event.target.value; setText(event.target.value); setNotice(''); }} placeholder="Start writing something good…" aria-describedby="privacy-note" /><div className="editor-footer"><span>{words} {words === 1 ? 'word' : 'words'}<span className="footer-divider">·</span>{text.length.toLocaleString()} / 20,000 characters</span><span className="saved-indicator"><span /> Only on this device</span></div></div>
        <aside className="review-pane" aria-label="Writing suggestions"><div className="review-heading"><div><h2>Your writing, refined<span aria-hidden="true">✧</span></h2><p>Small changes. More clarity.</p></div><span className="issue-count">{suggestions.length}</span></div><div className="checker-settings"><label className="toggle-label"><input type="checkbox" checked={useAI} onChange={(event) => setUseAI(event.target.checked)} /><span className="switch" aria-hidden="true" />Local AI<span className="experimental">EXPERIMENTAL</span></label><div className="backend-status" role="status"><span className={`status-dot ${status}`} />{status === 'checking' ? useAI ? 'Checking on your device…' : 'Checking local rules…' : status === 'error' ? 'Checker unavailable' : backendLabel}</div></div>
          {result?.modelError && current && <p className="model-warning">The local model is unavailable; rule suggestions still work. {result.modelError}</p>}
          {useAI && <p className="model-note">Tiny model trained on synthetic examples. Suggestions may miss real-world errors.</p>}
          <div className="suggestion-list" aria-live="polite" aria-busy={status === 'checking'}>
            {status === 'checking' ? <div className="empty-state"><span className="checking-symbol" aria-hidden="true">✧</span><p>Giving your words a look.</p><span>The first local AI check may take a moment.</span></div> : status === 'error' ? <div className="empty-state"><p>We couldn’t check this draft.</p><span>{error}</span></div> : suggestions.length ? suggestions.map((suggestion) => <SuggestionCard key={suggestion.id} suggestion={suggestion} onAccept={() => accept(suggestion)} onDismiss={() => setDismissed((previous) => new Set([...previous, suggestion.id]))} />) : <div className="empty-state"><span className="all-clear-symbol" aria-hidden="true">✓</span><p>No suggestions from this checker.</p><span>Keep writing in your own voice.</span></div>}
          </div><div className="review-footer"><button className="accept-all-button" disabled={!suggestions.length || !current} onClick={acceptAll}>Accept all suggestions <Arrow /></button></div>
        </aside>
      </section>
      <div className="beneath-editor"><p id="privacy-note"><svg width="15" height="17" viewBox="0 0 15 17" fill="none" aria-hidden="true"><path d="m7.5 1-6 2v5c0 4 6 7 6 7s6-3 6-7V3l-6-2Z" stroke="currentColor" strokeWidth="1.2" /><path d="m5 8 1.5 1.5L10 6" stroke="currentColor" strokeWidth="1.2" /></svg>Your text never leaves this tab. No accounts. No tracking.</p><span>Made for humans who write.</span></div>
      <div className="principles"><div><span aria-hidden="true">◎</span><p>Private by default</p><small>Checks run on your device.</small></div><div><span aria-hidden="true">⌘</span><p>Small model, open code</p><small>Understand what’s under the hood.</small></div><div><span aria-hidden="true">✳</span><p>Keep your voice</p><small>You choose what to change.</small></div></div>
    </main><footer className="site-footer"><Mark compact /><p>A little help. A lot of possibility.</p><span>Apache 2.0 · Open source</span></footer><div className="sr-only" role="status" aria-live="polite">{notice}</div>
  </div>;
}
