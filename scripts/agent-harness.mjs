import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { acquireLock, boundedInteger, changedProductPaths, git, hash, invokeCodex, jsonFile, productScopeAllowed, publicationAllowed, runCommand, runId, sourceStamp, validateReceipt, validateReview } from './agent-runtime.mjs';
import { planSchema, validatePlan } from './extension-driver.mjs';

export const reviewSchema = {type: 'object', additionalProperties: false, required: ['runId', 'evidenceHash', 'verdict', 'findings'], properties: {
  runId: {type: 'string'}, evidenceHash: {type: 'string'}, verdict: {type: 'string', enum: ['pass', 'fail']},
  findings: {type: 'array', items: {type: 'object', additionalProperties: false, required: ['kind', 'description'], properties: {
    kind: {type: 'string', enum: ['bug', 'limitation']}, description: {type: 'string'},
  }}},
}};
const codingSchema = {type: 'object', additionalProperties: false, required: ['title', 'summary'], properties: {title: {type: 'string'}, summary: {type: 'string'}}};
const discoverySchema = {type: 'object', additionalProperties: false, required: ['task', 'acceptance'], properties: {
  task: {type: 'string'}, acceptance: {type: 'array', items: {type: 'string'}},
}};

export function parseOptions(args) {
  const options = {mode: args.shift(), iterations: 2, codingModel: process.env.GAMMA_CODEX_MODEL ?? 'gpt-6-sol', publish: false, verifyOnly: false};
  if (!['e2e', 'loop'].includes(options.mode)) throw new Error('Usage: agent-harness.mjs e2e|loop [--task text] [--max-iterations 1..3] [--base ref] [--coding-model name] [--verify-only] [--publish]');
  while (args.length) {
    const flag = args.shift();
    if (flag === '--publish') options.publish = true;
    else if (flag === '--verify-only' || flag === '--no-code') options.verifyOnly = true;
    else if (['--task', '--base', '--coding-model', '--max-iterations', '--focus'].includes(flag)) {
      const value = args.shift();
      if (!value || value.startsWith('--')) throw new Error(`Missing value for ${flag}`);
      if (flag === '--max-iterations') options.iterations = boundedInteger(value, 'max-iterations', 3);
      else options[{'--task': 'task', '--base': 'base', '--coding-model': 'codingModel', '--focus': 'focus'}[flag]] = value;
    } else throw new Error(`Unknown option ${flag}`);
  }
  if (options.verifyOnly && (options.publish || options.task || options.base)) throw new Error('--verify-only cannot code, publish, or select a different base');
  if (options.mode === 'e2e' && (options.publish || options.verifyOnly || options.task || options.base)) throw new Error('Loop flags cannot be used with agent:e2e');
  return options;
}

export async function runAgentE2E({root, directory, build = true, focus = '', token}) {
  await mkdir(directory, {recursive: true, mode: 0o700});
  const id = runId();
  const stamp = await sourceStamp(root);
  const report = {runId: id, ...stamp, model: 'gpt-6-luna', reasoningEffort: 'max', passed: false, phase: 'build'};
  try {
    if (build) await runCommand('npm', ['run', 'build'], {cwd: root, log: join(directory, 'build.private.log')});
    report.phase = 'luna-plan';
    console.log('Luna-max: planning extension scenarios');
    const plan = validatePlan(await invokeCodex({root, directory, name: 'luna-plan', model: 'gpt-6-luna', schema: planSchema,
      prompt: `Focus requested by supervisor: ${focus || "Choose useful additional interaction cases."}\nYou drive E2E testing of Gamma EH, an offline Chrome extension. Produce a NEW bounded scenario plan, not a summary of existing results. Do not run browser/tools/commands. The deterministic controller executes your steps afterward on fictional localhost text only. Every scenario begins with enabled=true/useAI=false and a fresh fixture. IDs lowercase slugs. Choose 1-3 meaningful scenarios, <=48 actions total, <=20 per scenario. Include objective assertions.\nActions all have action,field,value,expected; unused arguments MUST be null. popup: field enabled/useAI, value boolean; focus/read: field draft/private/payment/optout/plain/rich; fill: field draft/plain, string value; accept/dismiss/close: all other arguments null; assertText: field and expected string; assertPanel: expected visible/hidden; assertBackend: expected rules/ai; staleAccept: field draft and identical string value/expected, used only after completed suggestion. assertBackend waits for completed analysis; use it before accept or dismiss. Do not fill rich/protected fields.\nInitial draft/private/payment/optout = "She have a freind."; plain = "She have a book."; rich = <strong>She have a book.</strong>. Rules correct freind->friend and sentence-initial She have->She has. AI is experimental. Rich DOM is preserved with unsupported notice. Protected fields must have no panel. An emoji prefix shifts UTF-16 offsets; "😀 A freind." should correct to "😀 A friend.". Dismiss removes only the first suggestion without changing text and keeps the panel visible, even when there are no remaining suggestions. Never assert a hidden panel immediately after dismiss. Closing the panel is a separate action. After closing a field, it stays paused until reload; do not assume resume semantics. Actual optional permission grant/revoke dialog is outside this suite; temporary localhost grant is static. Find additional useful sequences, e.g. toggle pause around a correction, clean text after an error, repeated edits, dismiss then focus another field. Avoid invented grammar expectations.\nSchema: ${JSON.stringify(planSchema)}`,
    }));
    await jsonFile(join(directory, 'plan.json'), plan);
    report.planHash = hash(plan);
    report.phase = 'browser';
    console.log('Browser: executing required checks and Luna scenarios');
    const browserDir = join(directory, 'browser');
    await mkdir(browserDir, {recursive: true, mode: 0o700});
    try {
      await runCommand(process.execPath, [fileURLToPath(new URL('./extension-driver.mjs', import.meta.url)), '--root', root, '--directory', browserDir,
        '--plan', join(directory, 'plan.json'), '--run-id', id], {cwd: root, timeoutMs: 300_000, log: join(directory, 'browser.private.log'),
        environment: token ? {GAMMA_AGENT_LOCK_TOKEN: token} : {}});
    } catch (error) {
      if (error.result?.code !== 1 || error.result?.timedOut || error.result?.interrupted) throw error;
      await readFile(join(browserDir, 'browser.json')); // Only a recorded assertion failure may proceed to review.
    } finally {
      for (const name of await readdir(browserDir)) if (name.startsWith('session-')) await rm(join(browserDir, name), {recursive: true, force: true});
    }
    const evidence = JSON.parse(await readFile(join(browserDir, 'browser.json'), 'utf8'));
    report.machinePassed = evidence.passed;
    // Review even failed browser runs, but no agent can waive machine failures.
    report.phase = 'luna-review';
    console.log('Luna-max: reviewing browser evidence and screenshots');
    const images = evidence.screenshots.filter(name => name === 'failure.png' || name.includes('ai-opt-in') || name.includes('utf16') || name.startsWith('luna-')).slice(-4).map(name => join(directory, 'browser', name));
    const review = await invokeCodex({root, directory, name: 'luna-review', model: 'gpt-6-luna', schema: reviewSchema, images,
      prompt: `Review this REAL browser execution and attached screenshots. Do not run tools or alter anything. Output runId exactly ${id}, evidenceHash exactly ${hash(evidence)}. Verdict pass requires evidence.passed=true and every required and planned scenario passing. Do not treat experimental synthetic model performance as real-world accuracy. Separate actual bugs from known limits; a bug requires verdict fail. Permission boundary is documented, not proof of native optional permission dialogs. Missing screenshots/actions/errors require scrutiny. Report concrete actionable defects with observed steps in description. Evidence:\n${JSON.stringify(evidence)}`,
    });
    validateReceipt(evidence, {id, stamp, plan});
    validateReview(review, evidence);
    if (JSON.stringify(await sourceStamp(root)) !== JSON.stringify(stamp)) throw new Error('Source changed after browser verification');
    report.evidenceHash = hash(evidence);
    report.phase = 'complete'; report.passed = true;
  } catch (error) {
    report.error = error.message;
    report.agentFailure = error.agentFailure === true || report.phase === 'luna-plan';
    report.interrupted = error.result?.interrupted === true;
    throw Object.assign(error, {report});
  } finally { await jsonFile(join(directory, 'report.json'), report); }
  return report;
}

async function clean(root) {
  if (await git(root, 'status', '--porcelain')) throw new Error('Refusing an uncommitted source checkout. Commit or stash your work first.');
}

function validateCodingResult(result) {
  if (!result || typeof result.summary !== 'string' || !result.summary.trim() || typeof result.title !== 'string'
    || !/^(fix|feat|refactor|test|docs)(\([a-z0-9-]+\))?: [^\r\n]{1,120}$/u.test(result.title)) throw new Error('Malformed coding result or non-Conventional Commit title');
  return result;
}

export async function runLoop({root, directory, options, token}) {
  const report = {runId: runId(), codingModel: options.codingModel, testerModel: 'gpt-6-luna', mode: options.verifyOnly ? 'verify-only' : 'coding',
    status: 'running', published: false, iterations: [], worktree: null};
  let worktree = root;
  let task = options.task;
  let codingResult;
  try {
    await clean(root);
    if (!options.verifyOnly) {
      let base;
      if (options.base) base = await git(root, 'rev-parse', '--verify', '--end-of-options', `${options.base}^{commit}`);
      else {
        await runCommand('git', ['fetch', 'origin'], {cwd: root, log: join(directory, 'fetch.private.log'), timeoutMs: 120_000});
        const defaultRef = await git(root, 'symbolic-ref', 'refs/remotes/origin/HEAD');
        base = await git(root, 'rev-parse', defaultRef);
      }
      const branch = `agent/gamma-${report.runId}`;
      worktree = join(directory, 'worktree');
      await git(root, 'worktree', 'add', '-b', branch, worktree, base);
      report.worktree = worktree; report.branch = branch; report.base = base;
      // npm ci creates real dependencies in this worktree. No shared symlink.
      await runCommand('npm', ['ci'], {cwd: worktree, log: join(directory, 'install.private.log'), timeoutMs: 600_000});
      if (!task) {
        const discovery = await invokeCodex({root: worktree, directory, name: 'discovery', model: options.codingModel, schema: discoverySchema,
          prompt: 'Read docs/roadmap.md and the product code. Select ONE smallest useful improvement or concrete reproducible bug in apps/ or packages/. Prefer a narrow user-visible correction/editing defect with a focused test. No model training, datasets, dependencies, remote services, or automation infrastructure. Do not edit files or run heavy commands. Return one task and objective acceptance criteria.'});
        if (typeof discovery.task !== 'string' || !discovery.task.trim() || !Array.isArray(discovery.acceptance) || !discovery.acceptance.length || discovery.acceptance.some(item => typeof item !== 'string')) throw new Error('Task discovery did not return a concrete task');
        task = discovery.task + '\nAcceptance:\n' + discovery.acceptance.join('\n');
      }
      report.task = task;
    }
    for (let iteration = 1; iteration <= (options.verifyOnly ? 1 : options.iterations); iteration++) {
      const attemptDir = join(directory, `iteration-${iteration}`);
      await mkdir(attemptDir, {recursive: true, mode: 0o700});
      const attempt = {number: iteration, gates: {}, passed: false};
      report.iterations.push(attempt);
      if (!options.verifyOnly) {
        console.log(`Iteration ${iteration}: Codex coding in isolated worktree`);
        const headBefore = await git(worktree, 'rev-parse', 'HEAD');
        const prior = report.iterations.at(-2);
        codingResult = validateCodingResult(await invokeCodex({root: worktree, directory: attemptDir, name: 'coding', model: options.codingModel, schema: codingSchema, writable: true, timeoutMs: 900_000,
          prompt: `You are the Codex coding agent in an isolated Gamma EH worktree. Implement one narrow task fully, with meaningful focused regression tests and relevant docs. Follow AGENTS.md. Typed text stays local. Do not run npm/install/build/lint/tests/browser/Codex or spawn agents: supervisor runs all gates sequentially. Do not commit, switch branches, push, create PRs, merge, release, deploy, change credentials, modify AGENTS.md, or weaken tests/gates. No dependencies/training/datasets/model changes. Edit only apps/, packages/, tests/, docs/ unless task explicitly requires another safe path. Return a Conventional Commit title and concise summary; no claims of tests you did not run.\nTask:\n${task}\n${prior ? `Previous gate failed. Inspect private logs and browser evidence under ${join(directory, `iteration-${iteration - 1}`)}. Failure summary: ${JSON.stringify(prior)}. Repair actual cause, preserving prior valid edits.` : ''}`,
        }));
        if (await git(worktree, 'rev-parse', 'HEAD') !== headBefore) throw new Error('Coding agent changed Git history');
        const changed = await git(worktree, 'status', '--porcelain');
        if (!productScopeAllowed(await changedProductPaths(worktree))) throw new Error('Coding agent edited outside the permitted product scope');
        attempt.changed = Boolean(changed);
        if (!changed) { report.status = 'no-op'; break; }
      }
      const stamp = await sourceStamp(worktree);
      attempt.source = stamp;
      try {
        for (const [name, args, timeoutMs] of [['check', ['run', 'check'], 900_000], ['browser', ['run', 'test:browser'], 300_000]]) {
          console.log(`Iteration ${iteration}: ${name} verification`);
          await runCommand('npm', args, {cwd: worktree, timeoutMs, log: join(attemptDir, `${name}.private.log`), environment: {GAMMA_AGENT_LOCK_TOKEN: token}});
          attempt.gates[name] = {passed: true, treeHash: stamp.treeHash};
        }
        const result = await runAgentE2E({root: worktree, directory: join(attemptDir, 'e2e'), build: false, focus: options.focus, token});
        attempt.gates.agent = {passed: result.passed, runId: result.runId, treeHash: result.treeHash};
        if (JSON.stringify(await sourceStamp(worktree)) !== JSON.stringify(stamp)) throw new Error('Source changed between validation gates');
        attempt.passed = true;
        report.status = options.verifyOnly ? 'verified' : 'changed-and-verified';
        break;
      } catch (error) {
        attempt.error = error.message;
        if (error.result?.interrupted || error.report?.interrupted) throw error;
        if (error.agentFailure || error.report?.agentFailure) throw error;
        if (error.result?.command === (process.env.GAMMA_CODEX_BIN ?? 'codex')) throw error;
        if (options.verifyOnly || iteration === options.iterations) throw error;
      }
    }
    const final = report.iterations.at(-1);
    if (options.publish && publicationAllowed(final?.gates ?? {}, final?.changed)) {
      if (JSON.stringify(await sourceStamp(worktree)) !== JSON.stringify(final.source)) throw new Error('Source changed before publication');
      // Publication is opt-in. Validation failure never reaches this block.
      await git(worktree, 'add', '--', 'apps', 'packages', 'tests', 'docs');
      await git(worktree, '-c', 'core.hooksPath=/dev/null', 'commit', '-m', codingResult.title);
      await runCommand('git', ['push', '-u', 'origin', report.branch], {cwd: worktree, timeoutMs: 120_000, log: join(directory, 'push.private.log')});
      const body = `## Summary\n\n${codingResult.summary}\n\n## Testing\n\nnpm run check; npm run test:browser; Luna-max planned extension E2E and evidence review all passed on the published source.\n\n## Follow-ups\n\nNone\n`;
      const bodyPath = join(directory, 'pr-body.md'); await writeFile(bodyPath, body, {mode: 0o600});
      await runCommand('gh', ['pr', 'create', '--draft', '--head', report.branch, '--title', codingResult.title, '--body-file', bodyPath], {cwd: worktree, timeoutMs: 120_000, log: join(directory, 'pr.private.log')});
      report.published = true;
    }
    if (report.status === 'running') throw new Error('Loop exhausted without a verified result');
  } catch (error) { report.status = error.result?.interrupted || error.report?.interrupted ? 'interrupted' : 'failed'; report.error = error.message; throw Object.assign(error, {report}); }
  finally { await jsonFile(join(directory, 'loop.json'), report); }
  return report;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = fileURLToPath(new URL('../', import.meta.url));
  let lock;
  let directory;
  try {
    const options = parseOptions(process.argv.slice(2));
    lock = await acquireLock(root);
    directory = join(root, 'artifacts/agents', `${options.mode}-${runId()}`);
    await mkdir(directory, {recursive: true, mode: 0o700});
    const result = options.mode === 'e2e' ? await runAgentE2E({root, directory, focus: options.focus, token: lock.token}) : await runLoop({root, directory, options, token: lock.token});
    console.log(JSON.stringify({directory, passed: result.passed ?? ['verified', 'changed-and-verified'].includes(result.status), status: result.status ?? result.phase, published: result.published ?? false}));
  } catch (error) { console.error(JSON.stringify({directory, passed: false, error: error.message})); process.exitCode = 1; }
  finally { await lock?.release(); }
}
