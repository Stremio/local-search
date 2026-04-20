'use strict';

// Thin Node.js wrapper around the wasm-bindgen output in ../pkg.
// The wasm module owns the index; this wrapper reshapes the flat score array
// returned by `search()` into `{doc, score}` and keeps the caller's original
// documents accessible by id.

const wasm = require('../pkg/localsearch.js');

class LocalSearch {
  constructor(handle, documents) {
    this._handle = handle;
    this._documents = documents;
  }

  /**
   * Search the index.
   * @param {string} query
   * @param {number} maxResults
   * @returns {Array<{doc: any, score: number}>}
   */
  search(query, maxResults) {
    const flat = this._handle.search(query, maxResults);
    const out = new Array(flat.length / 2);
    for (let i = 0, j = 0; i < flat.length; i += 2, j++) {
      out[j] = { doc: this._documents[flat[i]], score: flat[i + 1] };
    }
    return out;
  }

  /**
   * Prefix-search the indexed token set.
   * @param {string} queryToken
   * @param {number} maxResults
   * @returns {string[]}
   */
  autocomplete(queryToken, maxResults) {
    return this._handle.autocomplete(queryToken, maxResults);
  }

  get maxEditDistance() { return this._handle.maxEditDistance; }
  get maxEditDistanceBoost() { return this._handle.maxEditDistanceBoost; }
  get maxPrefixBoost() { return this._handle.maxPrefixBoost; }
  get scoreThreshold() { return this._handle.scoreThreshold; }

  /** Release the underlying wasm memory. Safe to call once. */
  free() {
    if (this._handle) {
      this._handle.free();
      this._handle = null;
      this._documents = null;
    }
  }
}

class LocalSearchBuilder {
  /**
   * @param {any[]} documents - arbitrary caller-side objects
   * @param {(doc: any) => string} [textExtractor] - returns the indexable text
   *   for a document. Defaults to the identity function (documents are strings).
   */
  constructor(documents, textExtractor) {
    if (!Array.isArray(documents)) {
      throw new TypeError('documents must be an array');
    }
    const extract = textExtractor || ((d) => d);
    const texts = new Array(documents.length);
    for (let i = 0; i < documents.length; i++) {
      const t = extract(documents[i]);
      if (typeof t !== 'string') {
        throw new TypeError(`textExtractor(documents[${i}]) did not return a string`);
      }
      texts[i] = t;
    }
    this._documents = documents;
    // Construct the wasm-side builder immediately so subsequent .boosts() etc.
    // can be chained through.
    this._inner = new wasm.LocalSearchBuilder(texts);
  }

  /**
   * @param {(doc: any) => number} boostFn
   */
  boostComputer(boostFn) {
    const boosts = new Float64Array(this._documents.length);
    for (let i = 0; i < this._documents.length; i++) {
      const b = boostFn(this._documents[i]);
      if (typeof b !== 'number' || !Number.isFinite(b)) {
        throw new TypeError(`boostComputer(documents[${i}]) must return a finite number`);
      }
      boosts[i] = b;
    }
    this._inner = this._inner.boosts(boosts);
    return this;
  }

  maxEditDistance(v) { this._inner = this._inner.maxEditDistance(v); return this; }
  maxEditDistanceBoost(v) { this._inner = this._inner.maxEditDistanceBoost(v); return this; }
  maxPrefixBoost(v) { this._inner = this._inner.maxPrefixBoost(v); return this; }
  scoreThreshold(v) { this._inner = this._inner.scoreThreshold(v); return this; }

  build() {
    const handle = this._inner.build();
    // _inner is consumed by build() on the Rust side; drop our reference too.
    this._inner = null;
    return new LocalSearch(handle, this._documents);
  }
}

module.exports = {
  LocalSearchBuilder,
  LocalSearch,
  defaultTokenize: wasm.defaultTokenize,
  defaults: {
    maxEditDistance: wasm.defaultMaxEditDistance(),
    maxEditDistanceBoost: wasm.defaultMaxEditDistanceBoost(),
    maxPrefixBoost: wasm.defaultMaxPrefixBoost(),
    scoreThreshold: wasm.defaultScoreThreshold(),
  },
};
