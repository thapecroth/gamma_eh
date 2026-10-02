import { useLayoutEffect, useRef, type ReactNode, type RefObject } from 'react';
import type { Suggestion } from '@gamma/engine';

export function WritingEditor({ text, suggestions, editorRef, onChange }: {
  text: string;
  suggestions: Suggestion[];
  editorRef: RefObject<HTMLTextAreaElement | null>;
  onChange: (text: string) => void;
}) {
  const layerRef = useRef<HTMLDivElement>(null);
  const mirrorRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const editor = editorRef.current;
    const layer = layerRef.current;
    const mirror = mirrorRef.current;
    if (!editor || !layer || !mirror) return;
    const synchronize = () => {
      // clientWidth excludes the scrollbar, so soft wraps match the textarea.
      layer.style.width = `${editor.clientWidth}px`;
      layer.style.height = `${editor.clientHeight}px`;
      mirror.style.transform = `translate(${-editor.scrollLeft}px, ${-editor.scrollTop}px)`;
    };
    synchronize();
    const observer = new ResizeObserver(synchronize);
    observer.observe(editor);
    editor.addEventListener('scroll', synchronize);
    return () => { observer.disconnect(); editor.removeEventListener('scroll', synchronize); };
  }, [editorRef, text]);

  const highlighted: ReactNode[] = [];
  let cursor = 0;
  for (const suggestion of [...suggestions].sort((a, b) => a.start - b.start || a.end - b.end)) {
    const { start, end, original, checkedText } = suggestion;
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < cursor || end <= start || end > text.length ||
      text.slice(start, end) !== original || (checkedText !== undefined && checkedText !== text)) continue;
    highlighted.push(text.slice(cursor, start));
    highlighted.push(<span key={suggestion.id} className="writing-error" data-start={start} data-end={end}>{original}</span>);
    cursor = end;
  }
  highlighted.push(text.slice(cursor));

  return <div className="writing-editor-container">
    <textarea id="writing-editor" ref={editorRef} className="writing-editor" spellCheck={false} maxLength={20_000}
      value={text} onChange={(event) => onChange(event.target.value)}
      placeholder="A blank page, a fresh start. Type or paste your own writing…" aria-describedby="privacy-note example-description" />
    <div ref={layerRef} className="writing-highlight-layer" aria-hidden="true">
      <div ref={mirrorRef} className="writing-highlights">{highlighted}{'\u200b'}</div>
    </div>
  </div>;
}
