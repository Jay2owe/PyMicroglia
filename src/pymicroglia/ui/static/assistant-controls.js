/* App-owned commands over existing controls; the shared ai-ify bridge runs them. */
(() => {
  'use strict';
  function visible(node) { return !!node.getClientRects().length && getComputedStyle(node).visibility !== 'hidden'; }
  function state() {
    const active = [...document.querySelectorAll('.view-tab.active,[data-workspace].active,[aria-current="page"],nav .active')].find(visible);
    return {page: location.pathname, view: active ? active.textContent.trim() : document.title};
  }
  function inspect() {
    const fields = [];
    for (const node of document.querySelectorAll('input,select,textarea')) {
      if (!visible(node) || node.closest('[data-agent="off"]') || node.type === 'password' || node.type === 'file') continue;
      const name = node.id || node.name || node.getAttribute('aria-label') || '';
      if (/password|secret|token|api.?key|credential/i.test(name)) continue;
      if (node.type === 'radio' && !node.checked) continue;
      fields.push({name, value: node.type === 'checkbox' ? node.checked : node.value});
      if (fields.length >= 100) break;
    }
    return {...state(), fields};
  }
  function install() {
    const api = window.aiify = window.aiify || {};
    if (api.setState) api.setState(state); else api._pendingState = state;
    const tools = [
      {name: 'workspace.inspect', description: 'Read the current visible view and its form values without running an analysis.',
       inputSchema: {type: 'object', properties: {}, additionalProperties: false}, execute: inspect},
      {name: 'workspace.navigate', description: 'Open an existing navigation tab by its exact visible label; native unsaved-work dialogs remain in force.',
       inputSchema: {type: 'object', properties: {label: {type: 'string'}}, required: ['label'], additionalProperties: false},
       execute: ({label}) => {
         const candidates = [...document.querySelectorAll('nav button,nav a,button[data-view],button[data-workspace]')];
         const button = candidates.find(node => visible(node) && !node.closest('[data-agent="off"]') && node.textContent.trim() === label);
         if (!button) throw new Error('Choose an existing visible navigation label.');
         button.click(); return state();
       }}
    ];
    for (const tool of tools) {
      if (api.registerTool) api.registerTool(tool); else (api._pendingTools ||= []).push(tool);
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install, {once: true});
  else install();
})();
