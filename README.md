# LocalSearch

`LocalSearch` is a client-side [full-text](https://en.wikipedia.org/wiki/Full-text_search) library.

- Written in [Rust](https://www.rust-lang.org/) and intended to run in [WASM](https://webassembly.org/).

- Able to index and search through 20 000 film titles in a few milliseconds.

- Leverages crates [fst](https://crates.io/crates/fst) and [fxhash](https://crates.io/crates/fxhash) for maximum performance.

- Tested with the Rust framework [Seed](https://seed-rs.org/) - see `/test_app`.

- Supports autocomplete.

- Integrates [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) & [Levenshtein](https://en.wikipedia.org/wiki/Levenshtein_distance) algorithms, edit distance & prefix boosting and some other technics to improve scoring.

---

### [Live Demo](https://stremio-search.netlify.app/)
- Is deployed manually to [Netlify](https://www.netlify.com/).

- It's the release build of `/test_app`.

- How to use it:
   1. Click the button `Download & Deserialize`. (The first download is usually slow because the dataset with 20k films isn't cached either by Netlify or by the browser.)
   1. Click `Index`
   1. Write a title name into the field `Query` - e.g. `office`

---

### Node.js

The Rust crate can be built as a WebAssembly module and used from Node.js with
the same performance as any other Rust-to-WASM target (no core logic is
reimplemented in JS).

Build and run the smoke test:

```sh
$ wasm-pack build --target nodejs --release --out-dir pkg
$ node nodejs/test.js
```

Usage:

```js
const { LocalSearchBuilder } = require('./nodejs');

const docs = [
  { id: 'tt0109830', name: 'Forrest Gump',    imdbRating: 8.8, popularity: 40 },
  { id: 'tt0137523', name: 'Fight Club',      imdbRating: 8.8, popularity: 35 },
  { id: 'tt0133093', name: 'The Matrix',      imdbRating: 8.7, popularity: 55 },
];

const maxPop = Math.max(...docs.map(d => d.popularity));
const search = new LocalSearchBuilder(docs, d => d.name)
  .boostComputer(d => Math.exp(d.imdbRating / 10) * Math.exp(d.popularity / maxPop))
  .maxEditDistance(1)
  .scoreThreshold(0.48)
  .build();

for (const { doc, score } of search.search('matrix', 5)) {
  console.log(score.toFixed(2), doc.name);
}

console.log(search.autocomplete('fo', 5)); // ['forrest']
```

Indicative timings on the 20 000-title Cinemeta dataset (Apple Silicon, Node 22):
build ~220 ms, `search()` ~0.7 ms/query, `autocomplete()` ~20 µs/query.

See [`nodejs/index.js`](nodejs/index.js) for the full wrapper and
[`nodejs/test.js`](nodejs/test.js) for correctness + benchmark tests.

#### IndexManager — server pattern for many lists × languages

[`nodejs/manager.js`](nodejs/manager.js) provides an `IndexManager` class
suited for Node servers that search across many lists in multiple languages:

- **English index per list, always hot** — built on demand or via `prewarmEnglish`.
- **User-language index per (list, lang), LRU-cached** by byte budget, rebuilt
  transparently when evicted.
- **Build coalescing** — concurrent queries for the same cold index share one
  build; the second caller doesn't trigger a duplicate rebuild.
- **Build concurrency limit** — caps the number of simultaneous cold builds so
  event-loop stalls stay bounded (default 2).
- **Automatic merge** across (list × language) results, deduplicated by item id
  with the best-scoring language winning.

```js
const { IndexManager } = require('./nodejs/manager.js');

const manager = new IndexManager({
  loadList: async (listId) => fetchListItemsFromDB(listId),
  titleFor: (item, lang) => item.titles[lang],            // or `item.titleEn` / `item.titleByLang[lang]`
  boostComputer: (item) => Math.exp(item.rating / 10),    // optional
  userLangBudgetBytes: 500 * 1024 * 1024,                 // LRU budget (default 500 MB)
  maxConcurrentBuilds: 2,                                 // default 2
  builderOptions: { maxEditDistance: 1, scoreThreshold: 0.48 },
});

// Warm the English indexes at server start.
await manager.prewarmEnglish(['movies-top', 'movies-90s', 'series-top', /* ... */]);

// Per-request query: 2 lists × 2 languages = 4 searches, merged by item id.
const results = await manager.search({
  query: 'matrix',
  listIds: ['movies-top', 'series-top'],
  userLang: 'fr',
  maxResults: 15,
});
// => [{ doc, score, matchedLang: 'en' | 'fr' }, ...]
```

Run the IndexManager tests (includes LRU unit tests + a 20 k-docs-across-10-lists
benchmark): `node nodejs/manager.test.js`.

##### Scaling notes

- **Single shared WASM linear memory.** All indexes in one Node process share
  the wasm32 heap (4 GB hard cap). Comfortable at hundreds of MB; if you
  approach 2 GB, shard across Node `worker_threads`, each with its own wasm
  module and its own subset of lists (searches are dispatched to the owning
  worker). Wasm instances can't be transferred between threads.
- **Cold builds block the event loop** during the wasm call. A 400 ms build on
  a 40 k-item list will stall other requests. For steady-state (warm) searches
  this is never an issue — they're ~1 ms. For cold builds on a busy server,
  running `IndexManager` inside dedicated `worker_thread`s (one per shard) is
  the recommended scale-up path.
- **Cluster / multi-process.** Every Node worker process keeps its own copy of
  every index — memory use is `processes × indexes`. Prefer vertical scaling
  of a single process with `worker_threads` over horizontal process forks.

### Development

Run unit and doc tests by `$ cargo test` from the project root. And then `$ cargo fmt --all`.

Please, test your changes manually in `/test_app` (see its README for more info). And don't forget to run `cargo make verify` in `/test_app` to format and lint the project before the push.

---

_built with love and serious coding skills by the Stremio Team_

<img src="https://blog.stremio.com/wp-content/uploads/2023/08/stremio-code-footer.jpg" width="300" />
