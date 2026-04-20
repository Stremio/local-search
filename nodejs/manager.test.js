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

// ---- Model (a): unified items (all language titles on one object) ----------

section('IndexManager – model (a) unified items');

const modelAlists = {
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
// Attach fake German titles to items with even ids, so "de" indexes are partial.
for (const list of Object.values(modelAlists)) {
  for (const it of list) {
    if (parseInt(it.id.slice(2), 10) % 2 === 0) {
      it.titleDe = it.titleEn.split('').reverse().join('');
    }
  }
}

function makeModelALoadListCounter() {
  const counts = new Map(); // `${id}:${lang}` -> count
  return {
    counts,
    loadList: async (id, _lang) => {
      const key = `${id}:${_lang}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
      await new Promise((r) => setImmediate(r));
      const list = modelAlists[id];
      if (!list) throw new Error('unknown list ' + id);
      return list; // same items regardless of lang (model a)
    },
  };
}

// 1. Basic English-only search across 2 lists with a shared item
{
  const { loadList, counts } = makeModelALoadListCounter();
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
  assert.strictEqual(counts.get('movies_top:en'), 1);
  assert.strictEqual(counts.get('movies_90s:en'), 1);
  mgr.free();
  console.log('  English-only search across 2 lists, dedup by id  ok');
}

// 2. English + user-language search finds items via localized title
{
  const { loadList } = makeModelALoadListCounter();
  const mgr = new IndexManager({
    loadList,
    titleFor: (it, lang) => (lang === 'en' ? it.titleEn : it.titleDe),
  });

  // "noitpecnI" is the reversed synthetic German title for "Inception" (tt2 — even id has titleDe).
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

// 3. User-lang with no translations → gracefully no-ops
{
  const { loadList } = makeModelALoadListCounter();
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

// 4. Build coalescing: concurrent searches for same cold key = 1 build
{
  const { loadList, counts } = makeModelALoadListCounter();
  const mgr = new IndexManager({
    loadList,
    titleFor: (it, lang) => (lang === 'en' ? it.titleEn : it.titleDe),
  });
  await Promise.all([
    mgr.search({ query: 'matrix', listIds: ['movies_top'], maxResults: 5 }),
    mgr.search({ query: 'fight',  listIds: ['movies_top'], maxResults: 5 }),
    mgr.search({ query: 'inception', listIds: ['movies_top'], maxResults: 5 }),
  ]);
  assert.strictEqual(counts.get('movies_top:en'), 1, 'loadList(movies_top, en) called once');
  assert.strictEqual(mgr.stats().englishBuilds, 1);
  mgr.free();
  console.log('  build coalescing: 3 concurrent cold searches → 1 build  ok');
}

// 5. LRU eviction of user-lang indexes
{
  const { loadList } = makeModelALoadListCounter();
  const mgr = new IndexManager({
    loadList,
    titleFor: (it, lang) => (lang === 'en' ? it.titleEn : it.titleDe),
    userLangBudgetBytes: 8 * 1024,
    estimateBytes: (items) => items.length * 5 * 1024,
  });
  await mgr.search({ query: 'matrix', listIds: ['movies_top'], userLang: 'de' });
  await mgr.search({ query: 'pulp',   listIds: ['movies_90s'], userLang: 'de' });
  const s = mgr.stats();
  assert.strictEqual(s.userLangBuilds, 2);
  assert.ok(s.userLangEvictions >= 1);
  await mgr.search({ query: 'matrix', listIds: ['movies_top'], userLang: 'de' });
  assert.ok(mgr.stats().userLangBuilds >= 3, 'evicted list rebuilds on re-access');
  mgr.free();
  console.log('  LRU evicts oldest user-lang index, rebuilds on re-access  ok');
}

// 6. prewarmEnglish
{
  const { loadList, counts } = makeModelALoadListCounter();
  const mgr = new IndexManager({
    loadList,
    titleFor: (it, lang) => (lang === 'en' ? it.titleEn : it.titleDe),
  });
  await mgr.prewarmEnglish(Object.keys(modelAlists));
  assert.strictEqual(mgr.stats().englishBuilds, 4);
  for (const id of Object.keys(modelAlists)) {
    assert.strictEqual(counts.get(`${id}:en`), 1);
  }
  await mgr.search({ query: 'matrix', listIds: ['movies_top'] });
  assert.strictEqual(mgr.stats().englishBuilds, 4, 'no additional builds');
  assert.ok(mgr.stats().englishHits >= 1);
  mgr.free();
  console.log('  prewarmEnglish builds all lists, no duplicate loads  ok');
}

// ---- Model (b): distinct per-language items (preferDoc / scoreCombine) -----

section('IndexManager – model (b) per-language items + preferDoc / scoreCombine');

// Per-language lists: the English and French lists hold *different* item
// objects with the same id but localized `name` fields.
const modelBlists = {
  'movies:en': [
    { id: 'tt1', name: 'The Matrix' },
    { id: 'tt2', name: 'Inception' },
    { id: 'tt3', name: 'Fight Club' },
    // tt5: titles share the "spiderman" token across languages.
    { id: 'tt5', name: 'Spider-Man Homecoming' },
  ],
  'movies:fr': [
    { id: 'tt1', name: 'Matrice' },
    { id: 'tt2', name: 'Origine' },         // "Inception" in fr, won't match "inception"
    // tt3 (Fight Club) — no French version
    { id: 'tt4', name: 'Le Parrain' },      // only in French (no English counterpart here)
    { id: 'tt5', name: 'Spider-Man Retour' }, // shares "spiderman" token with the English title
  ],
};

function modelBLoader() {
  return async (listId, lang) => modelBlists[`${listId}:${lang}`] ?? [];
}

// 7. preferDoc='preferUserLang' with cross-lang match returns the user-lang doc
{
  const mgr = new IndexManager({
    loadList: modelBLoader(),
    titleFor: (item, _lang) => item.name,  // model (b): item is already in the right lang
    preferDoc: 'preferUserLang',
  });
  // "matrix" token appears in both English ("The Matrix") and French ("Matrice")
  // — actually, "matrix" and "matrice" share only "matr" prefix, so with edit
  //  distance they'll both match tt1.
  const hits = await mgr.search({
    query: 'matrix',
    listIds: ['movies'],
    userLang: 'fr',
    maxResults: 5,
  });
  const tt1 = hits.find((h) => h.doc.id === 'tt1');
  assert.ok(tt1, 'tt1 found');
  assert.strictEqual(tt1.doc.name, 'Matrice', 'preferUserLang returns French doc object');
  mgr.free();
  console.log('  preferUserLang: cross-lang match returns user-lang doc  ok');
}

// 8. preferDoc='preferUserLang' with English-only match + user-lang translation exists
{
  const mgr = new IndexManager({
    loadList: modelBLoader(),
    titleFor: (item) => item.name,
    preferDoc: 'preferUserLang',
  });
  // "inception" matches only the English title (French is "Origine").
  // But tt2 DOES have a French doc. preferUserLang should still return it.
  const hits = await mgr.search({
    query: 'inception',
    listIds: ['movies'],
    userLang: 'fr',
    maxResults: 5,
  });
  const tt2 = hits.find((h) => h.doc.id === 'tt2');
  assert.ok(tt2, 'tt2 found via English match');
  assert.strictEqual(tt2.matchedLang, 'en', 'matched language is English');
  assert.strictEqual(tt2.doc.name, 'Origine',
    'preferUserLang returns French doc from side-map, even though query matched English');
  mgr.free();
  console.log('  preferUserLang: English-only match → French doc via side-map  ok');
}

// 9. preferDoc='preferUserLang' with English-only match + NO user-lang translation → English doc
{
  const mgr = new IndexManager({
    loadList: modelBLoader(),
    titleFor: (item) => item.name,
    preferDoc: 'preferUserLang',
  });
  // tt3 (Fight Club) has no French version.
  const hits = await mgr.search({
    query: 'fight',
    listIds: ['movies'],
    userLang: 'fr',
    maxResults: 5,
  });
  const tt3 = hits.find((h) => h.doc.id === 'tt3');
  assert.ok(tt3, 'tt3 found');
  assert.strictEqual(tt3.doc.name, 'Fight Club',
    'falls back to English doc when no French translation exists');
  mgr.free();
  console.log('  preferUserLang: English-only match + no translation → English doc fallback  ok');
}

// 10. preferDoc='scoreMax' (default) keeps current behavior: returns best-scoring variant
{
  const mgr = new IndexManager({
    loadList: modelBLoader(),
    titleFor: (item) => item.name,
    // preferDoc default is 'scoreMax'
  });
  const hits = await mgr.search({
    query: 'inception',
    listIds: ['movies'],
    userLang: 'fr',
    maxResults: 5,
  });
  const tt2 = hits.find((h) => h.doc.id === 'tt2');
  assert.ok(tt2, 'tt2 found');
  // scoreMax: only English matched "inception", so return English doc
  assert.strictEqual(tt2.doc.name, 'Inception');
  mgr.free();
  console.log('  preferDoc default (scoreMax) returns best-score variant  ok');
}

// 11. scoreCombine='sum' adds scores across languages for same id
{
  const mgrMax = new IndexManager({
    loadList: modelBLoader(),
    titleFor: (item) => item.name,
    scoreCombine: 'max', // default
  });
  const mgrSum = new IndexManager({
    loadList: modelBLoader(),
    titleFor: (item) => item.name,
    scoreCombine: 'sum',
  });
  // tt5 ("Spider-Man Homecoming" / "Spider-Man Retour") matches the "spider"
  // prefix in *both* language indexes — sum should exceed max.
  const hitsMax = await mgrMax.search({ query: 'spider', listIds: ['movies'], userLang: 'fr' });
  const hitsSum = await mgrSum.search({ query: 'spider', listIds: ['movies'], userLang: 'fr' });
  const tt5Max = hitsMax.find((h) => h.doc.id === 'tt5');
  const tt5Sum = hitsSum.find((h) => h.doc.id === 'tt5');
  assert.ok(tt5Max && tt5Sum, 'tt5 returned by both configs');
  assert.ok(tt5Sum.score > tt5Max.score,
    `sum (${tt5Sum.score}) should exceed max (${tt5Max.score}) when both langs match`);
  mgrMax.free();
  mgrSum.free();
  console.log('  scoreCombine=sum > scoreCombine=max when multiple langs match  ok');
}

// 12. Invalid option values throw
{
  assert.throws(
    () => new IndexManager({ loadList: () => [], titleFor: () => '', preferDoc: 'nope' }),
    /preferDoc/,
  );
  assert.throws(
    () => new IndexManager({ loadList: () => [], titleFor: () => '', scoreCombine: 'avg' }),
    /scoreCombine/,
  );
  console.log('  constructor rejects invalid preferDoc / scoreCombine  ok');
}

// ---- Realistic scale: 20k dataset split into 10 synthetic "lists" ----------

section('IndexManager – 20k dataset split into 10 lists');

{
  const dataPath = path.resolve(
    __dirname, '..', 'test_app', 'data', 'cinemeta_20_000_unformatted.json',
  );
  const records = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
  const CHUNK = Math.ceil(records.length / 10);
  const chunks = {};
  for (let i = 0; i < 10; i++) chunks['list' + i] = records.slice(i * CHUNK, (i + 1) * CHUNK);
  const maxPop = Math.max(...records.map((r) => r.popularity));

  const mgr = new IndexManager({
    loadList: async (id, _lang) => chunks[id],
    titleFor: (it, lang) => (lang === 'en' ? it.name : null),
    boostComputer: (r) => Math.exp(r.imdbRating / 10) * Math.exp(r.popularity / maxPop),
  });

  const t0 = process.hrtime.bigint();
  await mgr.prewarmEnglish(Object.keys(chunks));
  const prewarmMs = Number(process.hrtime.bigint() - t0) / 1e6;
  console.log('  prewarm 10 × ~2k = ' + records.length + ' docs: ' + prewarmMs.toFixed(1) + ' ms');

  const queries = ['office', 'matrix', 'godfather', 'avenger', 'harry'];
  const pairs = [['list0', 'list1'], ['list2', 'list3'], ['list4', 'list5']];
  for (const q of queries) for (const [a, b] of pairs) await mgr.search({ query: q, listIds: [a, b], maxResults: 10 });
  let totalNs = 0n;
  const runs = 20;
  for (let r = 0; r < runs; r++) {
    for (const q of queries) {
      for (const [a, b] of pairs) {
        const t = process.hrtime.bigint();
        await mgr.search({ query: q, listIds: [a, b], maxResults: 10 });
        totalNs += process.hrtime.bigint() - t;
      }
    }
  }
  const count = runs * queries.length * pairs.length;
  console.log('  avg search() across 2 lists: ' + (Number(totalNs) / 1e3 / count).toFixed(1) + ' µs');

  const hits = await mgr.search({ query: 'office', listIds: ['list0', 'list1'], maxResults: 5 });
  for (const h of hits) console.log('    ' + h.score.toFixed(2) + '  ' + h.doc.name + '  [' + h.matchedLang + ']');
  assert.ok(hits.length > 0);
  mgr.free();
}

console.log('\nAll IndexManager tests passed.');

} // end main()

main().catch((err) => { console.error(err); process.exit(1); });
