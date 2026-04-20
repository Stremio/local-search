'use strict';

// IndexManager: per-list English indexes always hot + LRU-cached per-(list, lang)
// user-language indexes, with build coalescing and a concurrency limiter.
//
// Single-threaded by design (see README for notes on scaling via worker_threads,
// which requires sharding indexes across workers because wasm instances cannot
// be transferred between threads).

const { LocalSearchBuilder } = require('./index.js');

// ---- small in-file LRU ------------------------------------------------------

class ByteSizedLRU {
  constructor({ maxBytes, onEvict }) {
    this.maxBytes = maxBytes;
    this.onEvict = onEvict;
    this._map = new Map(); // insertion-order = LRU order; re-inserted on touch
    this.bytes = 0;
  }

  get size() { return this._map.size; }

  get(key) {
    const entry = this._map.get(key);
    if (!entry) return undefined;
    // Touch: re-insert to move to newest.
    this._map.delete(key);
    this._map.set(key, entry);
    return entry.value;
  }

  set(key, value, bytes) {
    if (this._map.has(key)) {
      const old = this._map.get(key);
      this.bytes -= old.bytes;
      this._map.delete(key);
    }
    this._map.set(key, { value, bytes });
    this.bytes += bytes;
    this._evictUntilFits();
  }

  delete(key) {
    const entry = this._map.get(key);
    if (!entry) return;
    this._map.delete(key);
    this.bytes -= entry.bytes;
    if (this.onEvict) this.onEvict(key, entry.value);
  }

  clear() {
    for (const [k, { value }] of this._map) {
      if (this.onEvict) this.onEvict(k, value);
    }
    this._map.clear();
    this.bytes = 0;
  }

  _evictUntilFits() {
    while (this.bytes > this.maxBytes && this._map.size > 0) {
      const iter = this._map.entries().next();
      if (iter.done) break;
      const [key, { value, bytes }] = iter.value;
      this._map.delete(key);
      this.bytes -= bytes;
      if (this.onEvict) this.onEvict(key, value);
    }
  }
}

// ---- IndexManager -----------------------------------------------------------

/**
 * @typedef {Object} IndexManagerOptions
 * @property {(listId: string) => (any[] | Promise<any[]>)} loadList
 *   Async callback returning the items for a list. Called once per cold
 *   (list, lang) combination; items may be filtered internally per language.
 * @property {(item: any, lang: string) => (string | null | undefined)} titleFor
 *   Returns the indexable title for an item in the given language, or a falsy
 *   value if none exists (those items are skipped for that language's index).
 * @property {(item: any) => number} [boostComputer]
 *   Optional per-item boost, forwarded to the underlying builder.
 * @property {number} [userLangBudgetBytes=500*1024*1024]
 *   Memory budget for the LRU of user-language indexes.
 * @property {number} [maxConcurrentBuilds=2]
 *   Max number of concurrent cold builds. Keeps event-loop stalls predictable.
 * @property {(items: any[], lang: string) => number} [estimateBytes]
 *   Override for the LRU byte accounting. Default heuristic is ~1.5 KB/item.
 * @property {Object} [builderOptions]
 *   Options forwarded to every `LocalSearchBuilder`: `maxEditDistance`,
 *   `maxEditDistanceBoost`, `maxPrefixBoost`, `scoreThreshold`.
 */

class IndexManager {
  /** @param {IndexManagerOptions} opts */
  constructor(opts) {
    if (!opts || typeof opts.loadList !== 'function') {
      throw new TypeError('IndexManager: opts.loadList is required');
    }
    if (typeof opts.titleFor !== 'function') {
      throw new TypeError('IndexManager: opts.titleFor is required');
    }
    this.loadList = opts.loadList;
    this.titleFor = opts.titleFor;
    this.boostComputer = opts.boostComputer ?? null;
    this.builderOptions = opts.builderOptions ?? {};
    this.estimateBytes = opts.estimateBytes ?? ((items) => items.length * 1500);
    this.maxConcurrentBuilds = opts.maxConcurrentBuilds ?? 2;

    this._english = new Map(); // listId -> entry
    this._userLang = new ByteSizedLRU({
      maxBytes: opts.userLangBudgetBytes ?? 500 * 1024 * 1024,
      onEvict: (_key, entry) => entry.instance.free(),
    });

    // Shared raw items per list — avoids re-running `loadList` when building
    // additional language indexes for the same list while any index (English
    // or user-lang) is still referencing it.
    this._itemsCache = new Map(); // listId -> Promise<any[]>
    this._itemsRefs = new Map();  // listId -> refCount

    this._inFlight = new Map(); // key -> Promise<entry|null>

    this._buildSlots = this.maxConcurrentBuilds;
    this._buildWaiters = [];

    this._stats = {
      englishBuilds: 0,
      userLangBuilds: 0,
      englishHits: 0,
      userLangHits: 0,
      userLangMisses: 0,
      userLangEvictions: 0,
      searches: 0,
    };
    // Track evictions on the LRU.
    const origOnEvict = this._userLang.onEvict;
    this._userLang.onEvict = (k, v) => {
      this._stats.userLangEvictions++;
      this._releaseItems(v.listId);
      origOnEvict(k, v);
    };
  }

  stats() { return { ...this._stats, userLangBytes: this._userLang.bytes, userLangSize: this._userLang.size, englishSize: this._english.size }; }

  /**
   * Pre-build the English indexes for a set of lists.
   * Returns when all are ready.
   */
  async prewarmEnglish(listIds) {
    await Promise.all(listIds.map((id) => this._getEnglish(id)));
  }

  /**
   * Main query entry point.
   *
   * @param {Object} p
   * @param {string} p.query
   * @param {string[]} p.listIds
   * @param {string} [p.userLang]
   * @param {number} [p.maxResults=15]
   * @returns {Promise<Array<{doc: any, score: number, matchedLang: string}>>}
   */
  async search({ query, listIds, userLang, maxResults = 15 }) {
    this._stats.searches++;
    if (!Array.isArray(listIds) || listIds.length === 0) return [];

    const tasks = [];
    for (const listId of listIds) {
      tasks.push(
        this._getEnglish(listId).then((entry) => ({
          hits: entry.instance.search(query, maxResults),
          lang: 'en',
        })),
      );
      if (userLang && userLang !== 'en') {
        tasks.push(
          this._getUserLang(listId, userLang).then((entry) =>
            entry
              ? { hits: entry.instance.search(query, maxResults), lang: userLang }
              : null,
          ),
        );
      }
    }

    const parts = (await Promise.all(tasks)).filter(Boolean);
    return this._merge(parts, maxResults);
  }

  /** Free all indexes and clear caches. Idempotent. */
  free() {
    for (const entry of this._english.values()) entry.instance.free();
    this._english.clear();
    this._userLang.clear();
    this._itemsCache.clear();
    this._itemsRefs.clear();
  }

  // ---- internals ----------------------------------------------------------

  _merge(parts, n) {
    const best = new Map();
    for (const { hits, lang } of parts) {
      for (const hit of hits) {
        const id = hit.doc && hit.doc.id != null ? hit.doc.id : hit.doc;
        const cur = best.get(id);
        if (!cur || hit.score > cur.score) {
          best.set(id, { doc: hit.doc, score: hit.score, matchedLang: lang });
        }
      }
    }
    return [...best.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, n);
  }

  async _getEnglish(listId) {
    const cached = this._english.get(listId);
    if (cached) {
      this._stats.englishHits++;
      return cached;
    }
    return this._buildCoalesced(`en:${listId}`, async () => {
      const items = await this._acquireItems(listId);
      const entry = this._build(items, 'en', listId);
      this._english.set(listId, entry);
      this._stats.englishBuilds++;
      return entry;
    });
  }

  async _getUserLang(listId, lang) {
    const key = `${lang}:${listId}`;
    const cached = this._userLang.get(key);
    if (cached) {
      this._stats.userLangHits++;
      return cached;
    }
    return this._buildCoalesced(key, async () => {
      this._stats.userLangMisses++;
      const items = await this._acquireItems(listId);
      // Only items that actually have a title in `lang` are indexed.
      const filtered = items.filter((it) => {
        const t = this.titleFor(it, lang);
        return typeof t === 'string' && t.length > 0;
      });
      if (filtered.length === 0) {
        // Don't bother building — but we also don't hold items.
        this._releaseItems(listId);
        return null;
      }
      const entry = this._build(filtered, lang, listId);
      this._userLang.set(key, entry, entry.bytes);
      this._stats.userLangBuilds++;
      return entry;
    });
  }

  _build(items, lang, listId) {
    const builder = new LocalSearchBuilder(items, (it) => {
      const t = this.titleFor(it, lang);
      return typeof t === 'string' ? t : '';
    });
    if (this.boostComputer) builder.boostComputer(this.boostComputer);
    const bo = this.builderOptions;
    if (bo.maxEditDistance != null) builder.maxEditDistance(bo.maxEditDistance);
    if (bo.maxEditDistanceBoost != null) builder.maxEditDistanceBoost(bo.maxEditDistanceBoost);
    if (bo.maxPrefixBoost != null) builder.maxPrefixBoost(bo.maxPrefixBoost);
    if (bo.scoreThreshold != null) builder.scoreThreshold(bo.scoreThreshold);
    const instance = builder.build();
    return {
      instance,
      bytes: this.estimateBytes(items, lang),
      lang,
      listId,
      itemCount: items.length,
    };
  }

  async _acquireItems(listId) {
    this._itemsRefs.set(listId, (this._itemsRefs.get(listId) ?? 0) + 1);
    let p = this._itemsCache.get(listId);
    if (!p) {
      p = Promise.resolve().then(() => this.loadList(listId));
      this._itemsCache.set(listId, p);
    }
    return p;
  }

  _releaseItems(listId) {
    const n = (this._itemsRefs.get(listId) ?? 0) - 1;
    if (n <= 0) {
      this._itemsRefs.delete(listId);
      this._itemsCache.delete(listId);
    } else {
      this._itemsRefs.set(listId, n);
    }
  }

  async _buildCoalesced(key, fn) {
    const inFlight = this._inFlight.get(key);
    if (inFlight) return inFlight;
    const p = this._withBuildSlot(fn).finally(() => this._inFlight.delete(key));
    this._inFlight.set(key, p);
    return p;
  }

  async _withBuildSlot(fn) {
    while (this._buildSlots === 0) {
      await new Promise((resolve) => this._buildWaiters.push(resolve));
    }
    this._buildSlots--;
    try {
      return await fn();
    } finally {
      this._buildSlots++;
      const next = this._buildWaiters.shift();
      if (next) next();
    }
  }
}

module.exports = { IndexManager, ByteSizedLRU };
