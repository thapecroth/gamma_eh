// Inspect the extension's closed shadow root without exposing it to the page.
// UI invalidation can detach a DOM snapshot between CDP calls; re-read only
// that specific race and always release unsuccessful inspection sessions.
export async function extensionElement(context, page, selector) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const session = await context.newCDPSession(page);
    let retained = false;
    try {
      const {root} = await session.send('DOM.getDocument', {depth: -1, pierce: true});
      function findShadow(node) {
        if (node.attributes?.includes('data-gamma-ignore')) {
          const shadow = node.shadowRoots?.find(candidate => candidate.shadowRootType === 'closed');
          if (shadow) return shadow;
        }
        for (const child of [...node.children ?? [], ...node.shadowRoots ?? []]) {
          const found = findShadow(child);
          if (found) return found;
        }
      }
      const shadow = findShadow(root);
      if (!shadow) return null;
      const {nodeId} = await session.send('DOM.querySelector', {nodeId: shadow.nodeId, selector});
      if (!nodeId) return null;
      const {object} = await session.send('DOM.resolveNode', {nodeId});
      retained = true;
      return {session, objectId: object.objectId};
    } catch (error) {
      if (attempt === 2 || !/Could not find node with given id|No node with given id found/u.test(error.message)) throw error;
    } finally {
      if (!retained) await session.detach();
    }
  }
}
