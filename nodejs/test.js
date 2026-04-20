'use strict';

// Smoke test + micro-benchmark against the 20k Cinemeta dataset.
// Mirrors the boost formula used by /test_app/src/localsearch_search_panel.rs
// so we can cross-check results if needed.

const fs = require('fs');
const path = require('path');
const assert = require('assert');

const { LocalSearchBuilder, defaultTokenize, defaults } = require('./index.js');

function section(name) {
  console.log('\n=== ' + name + ' ===');
}

// -- 1. Tiny correctness tests (mirror the Rust unit tests) --

section('correctness');

{
  // tf_idf_test from src/lib.rs: "this is a a sample" / "this is another another example..."
  // is implicitly covered here; we rely on the wasm-side existing unit tests
  // (cargo test --lib passed on the native build). Here we exercise the public API.
  const records = [
    'apple',
    'this is a a sample',
    'this is another another example example example',
    'orange',
  ];
  const ls = new LocalSearchBuilder(records).build();
  const ac = ls.autocomplete('a', 5);
  assert.deepStrictEqual(ac, ['a', 'another', 'apple'], 'autocomplete("a")');
  ls.free();
  console.log('  autocomplete("a") -> [a, another, apple]  ok');
}

{
  const records = [
    'apple',
    'this is a a sample',
    'this is another another example example example',
    'another document',
    'orange',
  ];
  const ls = new LocalSearchBuilder(records).build();
  const results = ls.search('thi range', 5);
  const ids = results.map((r) => records.indexOf(r.doc));
  assert.deepStrictEqual(ids, [1, 2, 4], 'search("thi range")');
  ls.free();
  console.log('  search("thi range") -> [1, 2, 4]  ok');
}

{
  // defaultTokenize parity
  const tokens = defaultTokenize('Spider-Man: Far from Home');
  assert.deepStrictEqual(tokens, ['spiderman', 'far', 'from', 'home']);
  console.log('  defaultTokenize("Spider-Man: Far from Home") ok');
}

{
  // Defaults match the Rust constants.
  assert.strictEqual(defaults.maxEditDistance, 1);
  assert.strictEqual(defaults.maxEditDistanceBoost, 2);
  assert.strictEqual(defaults.maxPrefixBoost, 1.5);
  assert.strictEqual(Math.abs(defaults.scoreThreshold - 0.48) < 1e-12, true);
  console.log('  defaults match Rust constants  ok');
}

// -- 2. 20k dataset: build + search + autocomplete timing --

section('20k dataset');

const dataPath = path.resolve(
  __dirname,
  '..',
  'test_app',
  'data',
  'cinemeta_20_000_unformatted.json',
);
const records = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
console.log('  loaded ' + records.length + ' records');

const maxImdb = 10;
let maxPopularity = 0;
for (const r of records) {
  if (r.popularity > maxPopularity) maxPopularity = r.popularity;
}
const imdbWeight = 1;
const popWeight = 1;
const boost = (r) =>
  Math.exp((r.imdbRating / maxImdb) * imdbWeight) *
  Math.exp((r.popularity / maxPopularity) * popWeight);

const tBuildStart = process.hrtime.bigint();
const search = new LocalSearchBuilder(records, (r) => r.name)
  .boostComputer(boost)
  .build();
const tBuildEnd = process.hrtime.bigint();
const buildMs = Number(tBuildEnd - tBuildStart) / 1e6;
console.log('  build (index) took ' + buildMs.toFixed(1) + ' ms');

const queries = ['office', 'spider', 'godfather', 'matrix', 'avenger', 'dragon ball', 'harry potter', 'starwar'];

// Warm-up
for (const q of queries) search.search(q, 15);

let totalSearchNs = 0n;
const runs = 50;
for (let r = 0; r < runs; r++) {
  for (const q of queries) {
    const t0 = process.hrtime.bigint();
    search.search(q, 15);
    totalSearchNs += process.hrtime.bigint() - t0;
  }
}
const avgSearchUs = Number(totalSearchNs) / 1e3 / (runs * queries.length);
console.log(
  '  avg search() over ' + queries.length + ' queries x ' + runs + ' runs: ' +
  avgSearchUs.toFixed(1) + ' µs',
);

let totalAcNs = 0n;
const acPrefixes = ['o', 'of', 'off', 'offi', 'spid', 'god', 'dra', 'ave'];
for (const p of acPrefixes) search.autocomplete(p, 5);
for (let r = 0; r < runs; r++) {
  for (const p of acPrefixes) {
    const t0 = process.hrtime.bigint();
    search.autocomplete(p, 5);
    totalAcNs += process.hrtime.bigint() - t0;
  }
}
const avgAcUs = Number(totalAcNs) / 1e3 / (runs * acPrefixes.length);
console.log(
  '  avg autocomplete() over ' + acPrefixes.length + ' prefixes x ' + runs + ' runs: ' +
  avgAcUs.toFixed(1) + ' µs',
);

// Sample result sanity check
section('sample result');
const sample = search.search('office', 5);
for (const hit of sample) {
  console.log('  ' + hit.score.toFixed(3) + '  ' + hit.doc.name);
}
assert.ok(sample.length > 0, 'expected at least one result for "office"');
assert.ok(
  sample.some((h) => /office/i.test(h.doc.name)),
  'expected some result to contain "office"',
);

search.free();

console.log('\nAll tests passed.');
