'use strict';

// IndexManager tests. Uses synthetic multi-language titles on top of the real
// 20k Cinemeta dataset to exercise realistic sizes and shapes.

const fs = require('fs');
const path = require('path');
const assert = require('assert');

const { IndexManager, ByteSizedLRU } = require('./manager.js');

function section(name) { console.log('\n=== ' + name + ' ==='); }

async function main() {

// ---- LRU tests --------------------------------------------------------------

section('ByteSizedLRU');

{
  const evicted = [];
  const lru = new ByteSizedLRU({
    maxBytes: 100,
    onEvict: (k, v) => evicted.push([k, v]),
  });

  lru.set('a', 'A', 40);
  lru.set('b', 'B', 40);
  lru.set('c', 'C', 40); // total 120 > 100, evicts 'a'
  assert.deepStrictEqual(evicted, [['a', 'A']], 'evict oldest');

  // touch b -> now c is oldest
  assert.strictEqual(lru.get('b'), 'B');
  lru.set('d', 'D', 40); // evicts 'c'
  assert.deepStrictEqual(evicted, [['a', 'A'], ['c', 'C']], 'LRU ordering');

  lru.clear();
  assert.strictEqual(lru.size, 0);
  console.log('  LRU eviction + touch ordering + clear  ok');
}

// ---- IndexManager tests -----------------------------------------------------

section('IndexManager – synthetic multi-language');

// Build 4 "lists" of 3–5 movies each with English titles + fake German ("de")
// titles (reversed string, because we don't have real translations handy).
const lists = {
  movies_top: [
    { id: 'tt1', titleEn: 'The Matrix' },
    { id: 'tt2', titleEn: 'Inception' },
    { id: 'tt3', titleEn: 'Fight Club' },
  ],
  movies_90s: [
    { id: 'tt1', titleEn: 'The Matrix' }, // also in movies_top
    { id: 'tt4', titleEn: 'Pulp Fiction' },
    { id: 'tt5', titleEn: 'The Lion King' },
  ],
  series_top: [
    { id: 'tt10', titleEn: 'The Office' },
    { id: 'tt11', titleEn: 'Breaking Bad' },
  ],
  series_sci_fi: [
    { id: 'tt12', titleEn: 'Battlestar Galactica' },
    { id: 'tt13', titleEn: 'The Expanse' },
  ],
};
// Attach fake German titles to about half the items (so "de" indexes are smaller).
for (const list of Object.values(lists)) {
  for (const it of list) {
    if (parseInt(it.id.slice(2), 10) % 2 === 0) {
      it.titleDe = it.titleEn.split('').reverse().join('');
    }
  }
}

function makeLoadListCounter() {
  const counts = new Map();
  return {
    counts,
    loadList: async (id) => {
      counts.set(id, (counts.get(id) ?? 0) + 1);
      // Simulate async I/O
      await new Promise((r) => setImmediate(r));
      const list = lists[id];
      if (!list) throw new Error('unknown list ' + id);
      return list;
    },
  };
}

// 1. Basic English-only search
{
  const { loadList, counts } = makeLoadListCounter();
  const mgr = new IndexManager({
    loadList,
    titleFor: (it, lang) => (lang === 'en' ? it.titleEn : it.titleDe),
  });

  const hits = await mgr.search({
    query: 'matrix',
    listIds: ['movies_top', 'movies_90s'],
    maxResults: 5,
  });
  assert.ok(hits.length > 0, 'got results for "matrix"');
  assert.ok(hits.every((h) => h.matchedLang === 'en'), 'all English matches');
  // tt1 appears in both lists → merge should dedupe
  const ids = hits.map((h) => h.doc.id);
  assert.strictEqual(new Set(ids).size, ids.length, 'no duplicate ids across lists');
  assert.strictEqual(counts.get('movies_top'), 1, 'loadList called once per list');
  assert.strictEqual(counts.get('movies_90s'), 1);
  mgr.free();
  console.log('  English-only search across 2 lists, dedup by id  ok');
}

// 2. English + user-language search merges by best score
{
  const { loadList } = makeLoadListCounter();
  const mgr = new IndexManager({
    loadList,
    titleFor: (it, lang) => (lang === 'en' ? it.titleEn : it.titleDe),
  });

  // "noitpecnI" is the reversed synthetic German title for "Inception" (tt2 — even id has titleDe).
  // Matches only via the German index, not English.
  const hits = await mgr.search({
    query: 'noitpecnI',
    listIds: ['movies_top'],
    userLang: 'de',
    maxResults: 5,
  });
  assert.ok(hits.length > 0, 'got results for "noitpecnI"');
  assert.ok(hits.some((h) => h.matchedLang === 'de'), 'at least one German match');
  assert.ok(hits.some((h) => h.doc.id === 'tt2'), 'Inception found via German title');
  mgr.free();
  console.log('  user-lang search finds items with titleDe  ok');
}

// 3. User-lang with no translations → titleFor returns falsy → no error, no match
{
  const { loadList } = makeLoadListCounter();
  const mgr = new IndexManager({
    loadList,
    titleFor: (it, lang) => (lang === 'en' ? it.titleEn : null), // no translations
  });
  const hits = await mgr.search({
    query: 'matrix',
    listIds: ['movies_top'],
    userLang: 'fr',
    maxResults: 5,
  });
  assert.ok(hits.length > 0, 'English fallback still works');
  assert.ok(hits.every((h) => h.matchedLang === 'en'));
  mgr.free();
  console.log('  user-lang with no translations gracefully no-ops  ok');
}

// 4. Build coalescing: two concurrent searches for the same cold list = one build
{
  const { loadList, counts } = makeLoadListCounter();
  const mgr = new IndexManager({
    loadList,
    titleFor: (it, lang) => (lang === 'en' ? it.titleEn : it.titleDe),
  });

  await Promise.all([
    mgr.search({ query: 'matrix', listIds: ['movies_top'], maxResults: 5 }),
    mgr.search({ query: 'fight',  listIds: ['movies_top'], maxResults: 5 }),
    mgr.search({ query: 'inception', listIds: ['movies_top'], maxResults: 5 }),
  ]);
  assert.strictEqual(counts.get('movies_top'), 1, 'loadList called once despite 3 concurrent searches');
  const s = mgr.stats();
  assert.strictEqual(s.englishBuilds, 1);
  mgr.free();
  console.log('  build coalescing: 3 concurrent cold searches → 1 build  ok');
}

// 5. LRU eviction of user-lang indexes
{
  const { loadList } = makeLoadListCounter();
  // Tiny budget to force eviction after the 2nd language index.
  const mgr = new IndexManager({
    loadList,
    titleFor: (it, lang) => (lang === 'en' ? it.titleEn : it.titleDe),
    userLangBudgetBytes: 8 * 1024, // ~8 KB — enough for 1 small list, not 2
    estimateBytes: (items) => items.length * 5 * 1024, // force big apparent sizes
  });

  await mgr.search({ query: 'matrix', listIds: ['movies_top'], userLang: 'de' });
  await mgr.search({ query: 'pulp',   listIds: ['movies_90s'], userLang: 'de' });

  const s = mgr.stats();
  assert.strictEqual(s.userLangBuilds, 2, 'built 2 user-lang indexes');
  assert.ok(s.userLangEvictions >= 1, 'at least one eviction happened');

  // Re-query the evicted list → should rebuild.
  await mgr.search({ query: 'matrix', listIds: ['movies_top'], userLang: 'de' });
  assert.ok(mgr.stats().userLangBuilds >= 3, 'evicted list rebuilds on next access');
  mgr.free();
  console.log('  LRU evicts oldest user-lang index, rebuilds on re-access  ok');
}

// 6. prewarmEnglish
{
  const { loadList, counts } = makeLoadListCounter();
  const mgr = new IndexManager({
    loadList,
    titleFor: (it, lang) => (lang === 'en' ? it.titleEn : it.titleDe),
  });
  await mgr.prewarmEnglish(Object.keys(lists));
  assert.strictEqual(mgr.stats().englishBuilds, 4);
  assert.strictEqual(mgr.stats().englishSize, 4);
  for (const id of Object.keys(lists)) {
    assert.strictEqual(counts.get(id), 1, `loadList called once for ${id}`);
  }
  // Subsequent search hits the cache.
  await mgr.search({ query: 'matrix', listIds: ['movies_top'] });
  assert.strictEqual(mgr.stats().englishBuilds, 4, 'no additional builds');
  assert.ok(mgr.stats().englishHits >= 1);
  mgr.free();
  console.log('  prewarmEnglish builds all lists, no duplicate loads  ok');
}

// 7. Realistic scale: 20k dataset split into 10 synthetic "lists"
section('IndexManager – 20k dataset split into 10 lists');

{
  const dataPath = path.resolve(
    __dirname,
    '..',
    'test_app',
    'data',
    'cinemeta_20_000_unformatted.json',
  );
  const records = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
  const CHUNK = Math.ceil(records.length / 10);
  const chunks = {};
  for (let i = 0; i < 10; i++) {
    chunks['list' + i] = records.slice(i * CHUNK, (i + 1) * CHUNK);
  }
  const maxPop = Math.max(...records.map((r) => r.popularity));

  const mgr = new IndexManager({
    loadList: async (id) => chunks[id],
    titleFor: (it, lang) => (lang === 'en' ? it.name : null),
    boostComputer: (r) => Math.exp(r.imdbRating / 10) * Math.exp(r.popularity / maxPop),
  });

  const tPrewarm0 = process.hrtime.bigint();
  await mgr.prewarmEnglish(Object.keys(chunks));
  const prewarmMs = Number(process.hrtime.bigint() - tPrewarm0) / 1e6;
  console.log('  prewarm 10 × ~2k = ' + records.length + ' docs: ' + prewarmMs.toFixed(1) + ' ms');

  // Benchmark a batch of cross-list searches (2 lists each, like a real query).
  const queries = ['office', 'matrix', 'godfather', 'avenger', 'harry'];
  const pairs = [['list0', 'list1'], ['list2', 'list3'], ['list4', 'list5']];
  // Warm
  for (const q of queries) {
    for (const [a, b] of pairs) await mgr.search({ query: q, listIds: [a, b], maxResults: 10 });
  }
  let totalNs = 0n;
  const runs = 20;
  for (let r = 0; r < runs; r++) {
    for (const q of queries) {
      for (const [a, b] of pairs) {
        const t0 = process.hrtime.bigint();
        await mgr.search({ query: q, listIds: [a, b], maxResults: 10 });
        totalNs += process.hrtime.bigint() - t0;
      }
    }
  }
  const count = runs * queries.length * pairs.length;
  console.log('  avg search() across 2 lists: ' + (Number(totalNs) / 1e3 / count).toFixed(1) + ' µs');

  // Sanity check
  const hits = await mgr.search({ query: 'office', listIds: ['list0', 'list1'], maxResults: 5 });
  for (const h of hits) console.log('    ' + h.score.toFixed(2) + '  ' + h.doc.name + '  [' + h.matchedLang + ']');
  assert.ok(hits.length > 0);

  mgr.free();
}

console.log('\nAll IndexManager tests passed.');

} // end main()

main().catch((err) => { console.error(err); process.exit(1); });
