import assert from 'node:assert/strict';
import { join } from 'node:path';

export const inlineScenarioId = 'inline-interactions';

// Real pointer/keyboard events exercise the shipping content script. `inspect`
// uses CDP to read its closed shadow root without exposing it to the host page.
export async function verifyInline({fixture, directory, evidence, setting, text, wait, status, inspect}) {
  const rectangles = () => inspect('[aria-hidden="true"]', `function() { return Array.from(this.querySelectorAll('.underline')).map(node => { const r = node.getBoundingClientRect(); return {left:r.left,top:r.top,width:r.width,height:r.height}; }); }`);
  const cardVisible = () => inspect('.panel', 'function() { return !this.hidden; }');
  async function until(predicate, description) {
    const end = Date.now() + 5000;
    while (Date.now() < end) { if (await predicate()) return; await new Promise(ready => setTimeout(ready, 50)); }
    throw new Error(`Inline UI did not reach ${description}`);
  }
  async function hover(rect) { await fixture.mouse.move(rect.left + rect.width / 2, rect.top + rect.height / 2); }
  async function clickVisible(selector) {
    const rect = await inspect(selector, 'function() { const r=this.getBoundingClientRect(); return {x:r.left+r.width/2,y:r.top+r.height/2}; }');
    assert(rect, `Expected visible ${selector}`); await fixture.mouse.click(rect.x, rect.y);
  }
  const inline = {kind: 'mandatory', id: inlineScenarioId, passed: false, checks: []};
  evidence.scenarios.push(inline);
  await setting('enabled', true); await setting('useAI', false); await fixture.reload();
  await fixture.locator('#draft').focus();
  await wait(value => value?.includes('2 suggestions'), 'inline suggestions');
  assert.equal(await cardVisible(), false, 'No automatic floating panel');
  await fixture.screenshot({path: join(directory, 'inline-idle.png'), clip:{x:100,y:40,width:900,height:290}}); evidence.screenshots.push('inline-idle.png');
  let rects = await rectangles();
  assert.equal(rects.length, 2, 'Both incorrect words must be underlined');
  const expectedLeft = await fixture.locator('#draft').evaluate(field => {
    const style = getComputedStyle(field); const canvas = document.createElement('canvas'); const ctx = canvas.getContext('2d'); ctx.font = style.font;
    return field.getBoundingClientRect().left + field.clientLeft + parseFloat(style.paddingLeft) + ctx.measureText('She ').width;
  });
  assert(Math.abs(rects[0].left - expectedLeft) < 1.5, 'Underline must align with the incorrect word');
  await hover(rects[0]); await until(cardVisible, 'hover card');
  assert.match(await inspect('.body', 'function() { return this.textContent; }'), /have.*has/u);
  await fixture.screenshot({path: join(directory, 'inline-hover.png'), clip:{x:100,y:40,width:900,height:430}}); evidence.screenshots.push('inline-hover.png');
  await clickVisible('.accept');
  assert.equal(await text('draft'), 'She has a freind.', 'Pointer acceptance applies only the hovered correction');
  inline.checks.push('hover, pointer travel and accept, independent word alignment');
  await wait(value => value?.includes('1 suggestion'), 'remaining underline');
  rects = await rectangles(); await hover(rects[0]); await until(cardVisible, 'second hover');
  await fixture.mouse.move(20, 20); await until(async () => await cardVisible() === false, 'hover exit');
  await hover(rects[0]); await until(cardVisible, 'hover reentry');
  await fixture.mouse.move(20, 20); await until(async () => await cardVisible() === false, 'second hover exit');
  await fixture.locator('#draft').press('Alt+F8');
  assert.equal(await cardVisible(), true);
  assert.equal(await inspect('.accept', 'function() { return this.getRootNode().activeElement === this; }'), true, 'Keyboard opening focuses Accept');
  await fixture.keyboard.press('Escape');
  assert.equal(await cardVisible(), false); assert.equal(await fixture.locator('#draft').evaluate(field => document.activeElement === field), true);
  await hover(rects[0]); await until(cardVisible, 'dismiss card'); await clickVisible('.dismiss');
  assert.equal(await text('draft'), 'She has a freind.'); assert.equal((await rectangles()).length, 0);
  assert.equal(await cardVisible(), false);
  inline.checks.push('repeated hover exit, keyboard opening and Escape, dismiss without changing text');
  await fixture.locator('#draft').fill('😀 A freind.');
  await wait(value => value?.includes('1 suggestion'), 'emoji suggestion');
  rects = await rectangles(); await fixture.mouse.click(rects[0].left + rects[0].width / 2, rects[0].top + rects[0].height / 2);
  assert.equal(await fixture.locator('#draft').evaluate(field => document.activeElement === field), true, 'Clicking a word preserves editor focus');
  await until(cardVisible, 'clicked card');
  await fixture.locator('#draft').press('Alt+F8'); await fixture.keyboard.press('Enter');
  assert.equal(await text('draft'), '😀 A friend.'); inline.checks.push('click does not intercept caret, emoji offsets, keyboard accept');
  await fixture.locator('#draft').fill('A clean sentence.\n'.repeat(20) + 'A freind.');
  await fixture.locator('#draft').evaluate(field => { field.scrollTop = 0; });
  await wait(value => value?.includes('1 suggestion'), 'scrollable draft');
  await until(async () => (await rectangles())?.length === 0, 'offscreen underline clipped');
  await fixture.locator('#draft').evaluate(field => { field.scrollTop = field.scrollHeight; });
  await until(async () => (await rectangles())?.length === 1, 'scrolling reveals underline');
  rects = await rectangles(); const bottom = await fixture.locator('#draft').evaluate(field => field.getBoundingClientRect().bottom);
  assert(rects[0].top + rects[0].height <= bottom + 2);
  await hover(rects[0]); await until(cardVisible, 'scrolled hover'); await clickVisible('.accept');
  assert((await text('draft')).endsWith('A friend.')); inline.checks.push('textarea scroll and clipped offscreen marks');
  await fixture.locator('#draft').fill('A freind.');
  await wait(value => value?.includes('1 suggestion'), 'composition draft');
  await fixture.locator('#draft').dispatchEvent('compositionstart');
  assert.equal(await status(), null, 'IME composition clears stale marks');
  await fixture.locator('#draft').dispatchEvent('compositionend');
  await wait(value => value?.includes('1 suggestion'), 'composition ended');
  await fixture.locator('#draft').evaluate(field => { field.value = 'A changed sentence.'; });
  await fixture.mouse.move(200, 180);
  assert.equal(await status(), null, 'Programmatic text changes remove stale UI on interaction');
  inline.checks.push('IME and stale programmatic text');
  await fixture.locator('#draft').fill('A freind.');
  await wait(value => value?.includes('1 suggestion'), 'ignored-field boundary');
  await fixture.evaluate(() => { const ignored = document.createElement('textarea'); ignored.id = 'ignored'; ignored.setAttribute('data-gamma-ignore', ''); document.body.append(ignored); });
  await fixture.locator('#ignored').focus();
  assert.equal(await status(), null, 'A page-owned ignored field must clear the previous field UI');
  await fixture.locator('#draft').fill('A clean sentence. A freind.');
  await fixture.locator('#draft').evaluate(field => { field.style.width = '190px'; });
  await wait(value => value?.includes('1 suggestion'), 'wrapped field');
  rects = await rectangles(); assert.equal(rects.length, 1);
  const wrappedTop = rects[0].top;
  const firstLine = await fixture.locator('#draft').evaluate(field => field.getBoundingClientRect().top + field.clientTop + parseFloat(getComputedStyle(field).paddingTop));
  assert(wrappedTop > firstLine + 20, 'Wrapped word is marked on its second line');
  await fixture.locator('#draft').evaluate(field => { field.style.width = '100%'; });
  await until(async () => (await rectangles())?.[0]?.top < wrappedTop - 20, 'resize updates wrapping');
  const beforeMove = (await rectangles())[0];
  await fixture.evaluate(() => window.scrollBy(0, 60));
  await until(async () => Math.abs((await rectangles())?.[0]?.top - (beforeMove.top - 60)) < 1, 'page scroll alignment');
  inline.checks.push('ignored-field focus, line wrapping, resize and page scroll');
  await fixture.locator('#draft').evaluate(field => { field.style.width = '190px'; field.style.whiteSpace = 'pre'; field.scrollLeft = field.scrollWidth; });
  await until(async () => (await rectangles())?.length === 1, 'CSS unwrapped field');
  const unwrappedExpected = () => fixture.locator('#draft').evaluate(field => {
    const style = getComputedStyle(field); const canvas = document.createElement('canvas'); const ctx = canvas.getContext('2d'); ctx.font = style.font;
    ctx.fontKerning = style.fontKerning === 'none' ? 'none' : 'normal';
    return {left:field.getBoundingClientRect().left + field.clientLeft + parseFloat(style.paddingLeft) + ctx.measureText('A clean sentence. A ').width - field.scrollLeft,
      top:field.getBoundingClientRect().top + field.clientTop + parseFloat(style.paddingTop), lineHeight:parseFloat(style.lineHeight)};
  });
  await until(async () => { const expected=await unwrappedExpected(); const mark=(await rectangles())?.[0]; return mark && Math.abs(mark.left-expected.left)<1.5 && mark.top<expected.top+expected.lineHeight; }, 'unwrapped underline on the actual text line');
  await fixture.screenshot({path:join(directory,'inline-nowrap.png'),clip:{x:100,y:0,width:500,height:350}}); evidence.screenshots.push('inline-nowrap.png');
  await fixture.locator('#draft').evaluate(field => { field.style.width='100%'; field.style.whiteSpace=''; field.scrollLeft=0; });
  await until(async () => (await rectangles())?.[0]?.width > 0, 'restored field');
  const beforeTranslation=(await rectangles())[0];
  await fixture.locator('#draft').evaluate(field => { const parent=field.parentElement; parent.style.transition='transform 150ms linear'; parent.getBoundingClientRect(); parent.style.transform='translateY(64px)'; });
  await until(async () => Math.abs((await rectangles())?.[0]?.top-(beforeTranslation.top+64))<1, 'animated ancestor alignment');
  await fixture.locator('#draft').evaluate(field => { field.parentElement.style.transform='none'; });
  await until(async () => Math.abs((await rectangles())?.[0]?.top-beforeTranslation.top)<1, 'restored ancestor');
  await fixture.locator('#draft').press('Alt+F8');
  await fixture.locator('#draft').evaluate(field => {
    const parent=field.parentElement; parent.style.height='140px'; parent.style.overflow='auto';
    const spacer=document.createElement('div'); spacer.id='clip-spacer'; spacer.style.height='400px'; parent.append(spacer); parent.scrollTop=200;
  });
  await until(async () => await inspect('.badge','function() { return this.hidden; }') === true && await cardVisible() === false, 'fully clipped field hides its controls');
  assert.equal((await rectangles()).length,0);
  await fixture.locator('#draft').evaluate(field => {
    const parent=field.parentElement; parent.scrollTop=0; parent.style.height=''; parent.style.overflow=''; parent.style.transform=''; parent.style.transition=''; document.querySelector('#clip-spacer').remove();
  });
  await until(async () => (await rectangles())?.length === 1 && await inspect('.badge','function() { return this.hidden; }') === false, 'controls return with the editor');
  inline.checks.push('CSS nowrap and horizontal scroll, animated layout movement, clipped controls and restoration');
  await fixture.locator('#plain').evaluate(field => { field.replaceChildren(document.createTextNode('😀 A clean line.'), document.createElement('br'), document.createTextNode('A freind.')); });
  await fixture.evaluate(() => { document.documentElement.style.overflowY='scroll'; });
  await fixture.locator('#plain').focus();
  await fixture.locator('#plain').scrollIntoViewIfNeeded();
  await fixture.locator('#plain').evaluate(field => { window.scrollTo(0,field.getBoundingClientRect().top+window.scrollY-450); });
  await wait(value => value?.includes('1 suggestion'), 'plain line break');
  rects = await rectangles(); assert.equal(rects.length, 1);
  const nativeRect = await fixture.locator('#plain').evaluate(field => { const range = document.createRange(); range.setStart(field.lastChild, 2); range.setEnd(field.lastChild, 8); const rect = range.getBoundingClientRect(); return {left:rect.left,top:rect.top,width:rect.width}; });
  assert(Math.abs(rects[0].left - nativeRect.left) < 1 && Math.abs(rects[0].width - nativeRect.width) < 1 && Math.abs(rects[0].top - nativeRect.top) < 1);
  await hover(rects[0]); await until(cardVisible, 'plain hover'); await clickVisible('.accept');
  assert.equal(await fixture.locator('#plain').textContent(), '😀 A clean line.A friend.'); inline.checks.push('plain contenteditable Range alignment and BR offsets');
  assert.equal(await inspect('.badge','function() { return this.hidden; }'),false); inline.checks.push('document scrollbar and page scrolling keep visible editors marked');
  await fixture.evaluate(() => { document.documentElement.style.overflowY=''; });
  await fixture.setViewportSize({width: 390, height: 844});
  await fixture.locator('#draft').fill('A freind.');
  await wait(value => value?.includes('1 suggestion'), 'narrow viewport');
  await fixture.locator('#draft').press('Alt+F8');
  const cardBounds = await inspect('.panel', 'function() { const r=this.getBoundingClientRect(); return {left:r.left,right:r.right,top:r.top,bottom:r.bottom}; }');
  assert(cardBounds.left >= 12 && cardBounds.right <= 378 && cardBounds.top >= 12 && cardBounds.bottom <= 832, 'Card fits narrow viewport');
  await fixture.screenshot({path: join(directory, 'inline-mobile.png'), fullPage: true}); evidence.screenshots.push('inline-mobile.png');
  inline.checks.push('narrow viewport placement');
  await fixture.setViewportSize({width:1100,height:800}); await fixture.reload(); await fixture.locator('#draft').focus();
  await wait(value=>value?.includes('2 suggestions'),'pointer correction walkthrough');
  await hover((await rectangles())[0]); await until(cardVisible,'first correction'); await clickVisible('.accept');
  await wait(value=>value?.includes('1 suggestion'),'second correction');
  await hover((await rectangles())[0]); await until(cardVisible,'second correction'); await clickVisible('.accept');
  await wait(value=>value?.includes('No suggestions'),'clean draft');
  assert.equal(await text('draft'),'She has a friend.'); assert.equal((await rectangles()).length,0); assert.equal(await cardVisible(),false);
  await fixture.screenshot({path:join(directory,'inline-corrected.png'),clip:{x:100,y:40,width:900,height:290}}); evidence.screenshots.push('inline-corrected.png');
  inline.checks.push('complete pointer walkthrough and clean draft screenshot');
  await fixture.locator('#draft').fill('halo'); await wait(value=>value?.includes('1 suggestion'),'greeting suggestion');
  await hover((await rectangles())[0]); await until(cardVisible,'greeting hover');
  assert.match(await inspect('.body','function() { return this.textContent; }'),/Did you mean “hello” as a greeting/u);
  await fixture.screenshot({path:join(directory,'inline-greeting.png'),clip:{x:100,y:40,width:900,height:400}}); evidence.screenshots.push('inline-greeting.png');
  await clickVisible('.accept'); assert.equal(await text('draft'),'hello');
  inline.checks.push('original halo example: underline, hover explanation and pointer acceptance'); inline.passed = true;
}
