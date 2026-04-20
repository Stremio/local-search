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
 * @property {(listId: string, lang: string) => (any[] | Promise<any[]>)} loadList
 *   Async callback returning the items for one (list, language) pair. Called
 *   once per (listId, lang) combination that becomes hot; cached with
 *   reference counting while an index referencing it exists. Callers whose
 *   data source has all language titles on a single object can ignore `lang`
 *   and return the same items each time (they'll be re-read per language but
 *   the caller can memoize in a closure).
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
 * @property {'scoreMax' | 'preferUserLang'} [preferDoc='scoreMax']
 *   When the same item id matches in multiple languages, which `doc` object
 *   gets returned. `'scoreMax'` (default) returns the doc from whichever
 *   language scored highest — preserves the original behavior. `'preferUserLang'`
 *   returns the user-language doc whenever one exists for that item (even if
 *   the query matched only the English index), falling back to English if
 *   the item has no user-language version. Use `'preferUserLang'` when your
 *   per-language lists hold distinct doc objects with localized fields and
 *   you want result UIs to render in the user's language.
 * @property {'max' | 'sum'} [scoreCombine='max']
 *   How to combine scores when the same item id matches in multiple languages.
 *   `'max'` (default) keeps the best single-language score. `'sum'` adds
 *   scores from each matching language, rewarding items that matched in
 *   multiple languages.
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

    this.preferDoc = opts.preferDoc ?? 'scoreMax';
    if (this.preferDoc !== 'scoreMax' && this.preferDoc !== 'preferUserLang') {
      throw new TypeError(
        "IndexManager: preferDoc must be 'scoreMax' or 'preferUserLang'",
      );
    }
    this.scoreCombine = opts.scoreCombine ?? 'max';
    if (this.scoreCombine !== 'max' && this.scoreCombine !== 'sum') {
      throw new TypeError("IndexManager: scoreCombine must be 'max' or 'sum'");
    }

    this._english = new Map(); // listId -> entry
    this._userLang = new ByteSizedLRU({
      maxBytes: opts.userLangBudgetBytes ?? 500 * 1024 * 1024,
      onEvict: (_key, entry) => entry.instance.free(),
    });

    // Raw items cache, keyed per (listId, lang). Reference-counted by the
    // number of indexes (English + user-lang) currently referencing each key.
    this._itemsCache = new Map(); // `${listId}:${lang}` -> Promise<any[]>
    this._itemsRefs = new Map();  // same key -> refCount

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
      this._releaseItems(v.listId, v.lang);
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

    const useUser = userLang && userLang !== 'en';
    const enPromises = listIds.map((id) => this._getEnglish(id));
    const userPromises = useUser
      ? listIds.map((id) => this._getUserLang(id, userLang))
      : [];

    const [enEntries, userEntries] = await Promise.all([
      Promise.all(enPromises),
      Promise.all(userPromises),
    ]);

    const parts = [];
    for (const e of enEntries) {
      parts.push({ hits: e.instance.search(query, maxResults), lang: 'en' });
    }
    // `userLangSideMaps` is consulted by `preferUserLang` to look up the
    // localized doc for items that only matched in the English index but
    // have a user-language version available.
    const userLangSideMaps = [];
    for (const e of userEntries) {
      if (!e) continue;
      parts.push({ hits: e.instance.search(query, maxResults), lang: userLang });
      userLangSideMaps.push(e.idToDoc);
    }

    return this._merge(parts, userLangSideMaps, userLang, maxResults);
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

  _merge(parts, userLangSideMaps, userLang, n) {
    // Aggregate hits per item id: collect the doc variant seen in each
    // matching language, plus per-language scores so we can apply the
    // `scoreCombine` and `preferDoc` policies at output time.
    const agg = new Map();
    for (const { hits, lang } of parts) {
      for (const hit of hits) {
        const id = hit.doc && hit.doc.id != null ? hit.doc.id : hit.doc;
        let e = agg.get(id);
        if (!e) {
          e = {
            scoreMax: hit.score,
            scoreSum: hit.score,
            bestLang: lang,
            docsByLang: new Map([[lang, hit.doc]]),
          };
          agg.set(id, e);
        } else {
          e.scoreSum += hit.score;
          if (hit.score > e.scoreMax) {
            e.scoreMax = hit.score;
            e.bestLang = lang;
          }
          e.docsByLang.set(lang, hit.doc);
        }
      }
    }

    const preferUser =
      this.preferDoc === 'preferUserLang' && userLang && userLang !== 'en';

    const results = [];
    for (const [id, e] of agg) {
      const score = this.scoreCombine === 'sum' ? e.scoreSum : e.scoreMax;

      let doc;
      if (preferUser) {
        // 1. Prefer the user-language hit when the query matched it directly.
        if (e.docsByLang.has(userLang)) {
          doc = e.docsByLang.get(userLang);
        } else {
          // 2. Else look up the user-language variant in any loaded user-lang
          //    side-map (the query didn't match it, but the translation exists).
          let fromSide;
          for (const sm of userLangSideMaps) {
            if (sm && sm.has(id)) {
              fromSide = sm.get(id);
              break;
            }
          }
          // 3. Fall back to the doc from whichever language did match.
          doc = fromSide ?? e.docsByLang.get(e.bestLang);
        }
      } else {
        doc = e.docsByLang.get(e.bestLang);
      }

      results.push({ doc, score, matchedLang: e.bestLang });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, n);
  }

  async _getEnglish(listId) {
    const cached = this._english.get(listId);
    if (cached) {
      this._stats.englishHits++;
      return cached;
    }
    return this._buildCoalesced(`en:${listId}`, async () => {
      const items = await this._acquireItems(listId, 'en');
      // Filter to items that actually have an English title.
      const filtered = items.filter((it) => {
        const t = this.titleFor(it, 'en');
        return typeof t === 'string' && t.length > 0;
      });
      const entry = this._build(filtered, 'en', listId);
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
      const items = await this._acquireItems(listId, lang);
      // Only items that actually have a title in `lang` are indexed.
      const filtered = items.filter((it) => {
        const t = this.titleFor(it, lang);
        return typeof t === 'string' && t.length > 0;
      });
      if (filtered.length === 0) {
        // Don't bother building — but we also don't hold items.
        this._releaseItems(listId, lang);
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

    const entry = {
      instance,
      bytes: this.estimateBytes(items, lang),
      lang,
      listId,
      itemCount: items.length,
    };
    // For non-English indexes, keep an id → doc side-map so `preferUserLang`
    // can return the localized doc even for items whose query match only
    // fired on the English index.
    if (lang !== 'en') {
      const idToDoc = new Map();
      for (const item of items) {
        const id = item && item.id != null ? item.id : item;
        idToDoc.set(id, item);
      }
      entry.idToDoc = idToDoc;
    }
    return entry;
  }

  async _acquireItems(listId, lang) {
    const key = `${listId}:${lang}`;
    this._itemsRefs.set(key, (this._itemsRefs.get(key) ?? 0) + 1);
    let p = this._itemsCache.get(key);
    if (!p) {
      p = Promise.resolve().then(() => this.loadList(listId, lang));
      this._itemsCache.set(key, p);
    }
    return p;
  }

  _releaseItems(listId, lang) {
    const key = `${listId}:${lang}`;
    const n = (this._itemsRefs.get(key) ?? 0) - 1;
    if (n <= 0) {
      this._itemsRefs.delete(key);
      this._itemsCache.delete(key);
    } else {
      this._itemsRefs.set(key, n);
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
