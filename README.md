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

### Development

Run unit and doc tests by `$ cargo test` from the project root. And then `$ cargo fmt --all`.

Please, test your changes manually in `/test_app` (see its README for more info). And don't forget to run `cargo make verify` in `/test_app` to format and lint the project before the push.

---

_built with love and serious coding skills by the Stremio Team_

<img src="https://blog.stremio.com/wp-content/uploads/2018/03/new-logo-cat-blog.jpg" width="300" />
