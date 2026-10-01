// Test-only bootstrap: instrument the actual built worker before importing it.
// The shipped extension and web worker never modify browser globals.
const parameters = new URL(self.location.href).searchParams;
Object.defineProperty(navigator, 'userAgentData', {value: {platform: 'Windows'}});
const requestAdapter = navigator.gpu?.requestAdapter.bind(navigator.gpu);
const record = async options => {
  self.postMessage({adapterRequest: {hasPowerPreference: options != null && 'powerPreference' in options, powerPreference: options?.powerPreference}});
  return parameters.get('mode') === 'unavailable' ? null : await requestAdapter?.(options) ?? null;
};
if (navigator.gpu) navigator.gpu.requestAdapter = record;
else Object.defineProperty(navigator, 'gpu', {value: {requestAdapter: record}});
await import(parameters.get('module'));
self.postMessage({ready: true});
