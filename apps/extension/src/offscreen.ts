import { isCheckMessage, type CheckResponse } from './protocol';
import { checkLocally } from './local-checker';

chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse: (response: CheckResponse) => void) => {
  if (!isCheckMessage(message) || message.target !== 'offscreen' || sender.id !== chrome.runtime.id || sender.tab) return;
  void checkLocally(message).then(sendResponse).catch((error: unknown) => {
    sendResponse({requestId: message.requestId, error: error instanceof Error ? error.message : 'The local checker could not start.'});
  });
  return true;
});
