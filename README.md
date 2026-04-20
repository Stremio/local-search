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

### Development

Run unit and doc tests by `$ cargo test` from the project root. And then `$ cargo fmt --all`.

Please, test your changes manually in `/test_app` (see its README for more info). And don't forget to run `cargo make verify` in `/test_app` to format and lint the project before the push.

---

_built with love and serious coding skills by the Stremio Team_

<img src="https://blog.stremio.com/wp-content/uploads/2023/08/stremio-code-footer.jpg" width="300" />
