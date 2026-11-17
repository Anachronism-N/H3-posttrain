// Local adapter for an absent Artifact Tool standard-font import map.
// Uses the bundled pdf-lib metrics; does not change the installed runtime.
const Module = require('node:module');
const path = require('node:path');
const runtime = process.env.ARTIFACT_NODE_MODULES || 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-611-0_26.1112.611.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules';
const {Font} = require(path.join(runtime, '@pdf-lib/standard-fonts/lib/Font.js'));
const original = Module._load;
const cache = new Map();
Module._load = function(request, parent, isMain) {
  if (!request.startsWith('#standard-fonts/')) return original.apply(this, arguments);
  if (cache.has(request)) return cache.get(request);
  const key = request.split('/')[1];
  const name = key.replace(/^(Courier|Helvetica)(Bold|Oblique)/, '$1-$2')
    .replace(/^Times(Roman|Bold|Italic)/, 'Times-$1');
  const f = Font.load(name), names = f.CharMetrics.map(c => c.N);
  const indexes = new Map(names.map((n, i) => [n, i]));
  const groups = new Map();
  for (const [a,b,k] of f.KernPairs) {
    if (!indexes.has(a) || !indexes.has(b)) continue;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(indexes.get(a)*names.length + indexes.get(b));
  }
  const kernPairs = [];
  for (const [k, values] of groups) {
    values.sort((a,b)=>a-b); let previous = 0;
    kernPairs.push(k, values.map(v => {const delta=v-previous;previous=v;return delta;}));
  }
  const result = {name, bbox:f.FontBBox, ascender:f.Ascender || 0,
    descender:f.Descender || 0, xHeight:f.XHeight || 0, capHeight:f.CapHeight || 0,
    glyphNames:names.join(' '), glyphWidths:f.CharMetrics.map(c=>c.WX), kernPairs};
  cache.set(request,result); return result;
};
