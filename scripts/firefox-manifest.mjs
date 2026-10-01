// Derive shared metadata/permissions so browser variants cannot drift.
export function firefoxManifest(chromeManifest) {
  const {minimum_chrome_version, background, ...shared} = chromeManifest;
  return {...shared,
    permissions: shared.permissions?.filter(permission => permission !== 'offscreen'),
    background: {scripts: ['background.js']},
    browser_specific_settings: {gecko: {
      id: 'gamma-eh@extensions.gamma-eh.local',
      strict_min_version: '140.0',
      data_collection_permissions: {required: ['none']},
    }, gecko_android: {strict_min_version: '142.0'}},
  };
}
