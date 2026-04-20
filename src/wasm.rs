//! wasm-bindgen layer exposing `LocalSearch` to JavaScript (Node.js).
//!
//! Design notes:
//! - Documents are provided as pre-extracted text strings (`string[]`). The
//!   caller keeps the original objects on the JS side and uses returned indices
//!   to resolve them. This keeps the hot search path entirely inside WASM with
//!   no cross-boundary callbacks.
//! - Boosts are provided as an optional `Float64Array`-compatible `Vec<f64>`
//!   aligned by index with the documents.
//! - `DefaultTokenizer` is always used internally; a standalone
//!   `default_tokenize` helper is exported for parity on the JS side.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

use crate::{
    DefaultTokenizer, LocalSearch, LocalSearchBuilder as InnerBuilder, Tokenizer,
    DEFAULT_MAX_EDIT_DISTANCE, DEFAULT_MAX_EDIT_DISTANCE_BOOST, DEFAULT_MAX_PREFIX_BOOST,
    DEFAULT_SCORE_THRESHOLD,
};

/// Builder for `LocalSearch`. Stashes configuration until `.build()` is called.
///
/// Exposed to JS as `LocalSearchBuilder`.
#[wasm_bindgen]
pub struct LocalSearchBuilder {
    texts: Vec<String>,
    boosts: Option<Vec<f64>>,
    max_edit_distance: Option<usize>,
    max_edit_distance_boost: Option<f64>,
    max_prefix_boost: Option<f64>,
    score_threshold: Option<f64>,
}

#[wasm_bindgen]
impl LocalSearchBuilder {
    /// `texts[i]` is the indexable text for document `i`.
    #[wasm_bindgen(constructor)]
    pub fn new(texts: Vec<JsValue>) -> Result<LocalSearchBuilder, JsValue> {
        let texts = texts
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_string()
                    .ok_or_else(|| JsValue::from_str(&format!("texts[{i}] is not a string")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            texts,
            boosts: None,
            max_edit_distance: None,
            max_edit_distance_boost: None,
            max_prefix_boost: None,
            score_threshold: None,
        })
    }

    /// Per-document boosts aligned by index with `texts`. Length must match.
    pub fn boosts(mut self, boosts: Vec<f64>) -> Result<LocalSearchBuilder, JsValue> {
        if boosts.len() != self.texts.len() {
            return Err(JsValue::from_str(&format!(
                "boosts length {} != texts length {}",
                boosts.len(),
                self.texts.len()
            )));
        }
        self.boosts = Some(boosts);
        Ok(self)
    }

    #[wasm_bindgen(js_name = maxEditDistance)]
    pub fn max_edit_distance(mut self, value: usize) -> LocalSearchBuilder {
        self.max_edit_distance = Some(value);
        self
    }

    #[wasm_bindgen(js_name = maxEditDistanceBoost)]
    pub fn max_edit_distance_boost(mut self, value: f64) -> LocalSearchBuilder {
        self.max_edit_distance_boost = Some(value);
        self
    }

    #[wasm_bindgen(js_name = maxPrefixBoost)]
    pub fn max_prefix_boost(mut self, value: f64) -> LocalSearchBuilder {
        self.max_prefix_boost = Some(value);
        self
    }

    #[wasm_bindgen(js_name = scoreThreshold)]
    pub fn score_threshold(mut self, value: f64) -> LocalSearchBuilder {
        self.score_threshold = Some(value);
        self
    }

    pub fn build(self) -> LocalSearchHandle {
        // Doc type is `IndexedText { id, text }`: the id preserves the caller's
        // original array position for result lookup, and the text is owned by
        // the doc so the `text_extractor` closure can return a &str with a
        // lifetime tied to the doc reference (as the `Fn(&T) -> &str` bound
        // requires).
        let docs: Vec<IndexedText> = self
            .texts
            .into_iter()
            .enumerate()
            .map(|(id, text)| IndexedText { id, text })
            .collect();
        let boosts = self.boosts;

        let mut builder: InnerBuilder<IndexedText, DefaultTokenizer> =
            InnerBuilder::new(docs, |d: &IndexedText| d.text.as_str());

        if let Some(bs) = boosts {
            builder = builder.boost_computer(move |d: &IndexedText| bs[d.id]);
        }
        if let Some(v) = self.max_edit_distance {
            builder = builder.max_edit_distance(v);
        }
        if let Some(v) = self.max_edit_distance_boost {
            builder = builder.max_edit_distance_boost(v);
        }
        if let Some(v) = self.max_prefix_boost {
            builder = builder.max_prefix_boost(v);
        }
        if let Some(v) = self.score_threshold {
            builder = builder.score_threshold(v);
        }

        LocalSearchHandle {
            inner: builder.build(),
        }
    }
}

struct IndexedText {
    id: usize,
    text: String,
}

/// A built `LocalSearch` index. Exposed to JS as `LocalSearch`.
#[wasm_bindgen(js_name = LocalSearch)]
pub struct LocalSearchHandle {
    inner: LocalSearch<IndexedText, DefaultTokenizer>,
}

#[wasm_bindgen(js_class = LocalSearch)]
impl LocalSearchHandle {
    /// Returns a flat `Float64Array`-compatible `Vec<f64>` laid out as
    /// `[id0, score0, id1, score1, ...]` to avoid allocating N small JS objects.
    /// The JS wrapper reshapes this into `[{id, score}, ...]`.
    pub fn search(&self, query: &str, max_results: usize) -> Vec<f64> {
        let hits = self.inner.search(query, max_results);
        let mut out = Vec::with_capacity(hits.len() * 2);
        for (doc, score) in hits {
            out.push(doc.id as f64);
            out.push(score);
        }
        out
    }

    pub fn autocomplete(&self, query_token: &str, max_results: usize) -> Vec<JsValue> {
        self.inner
            .autocomplete(query_token, max_results)
            .into_iter()
            .map(JsValue::from)
            .collect()
    }

    #[wasm_bindgen(getter, js_name = maxEditDistance)]
    pub fn max_edit_distance(&self) -> usize {
        self.inner.max_edit_distance
    }

    #[wasm_bindgen(getter, js_name = maxEditDistanceBoost)]
    pub fn max_edit_distance_boost(&self) -> f64 {
        self.inner.max_edit_distance_boost
    }

    #[wasm_bindgen(getter, js_name = maxPrefixBoost)]
    pub fn max_prefix_boost(&self) -> f64 {
        self.inner.max_prefix_boost
    }

    #[wasm_bindgen(getter, js_name = scoreThreshold)]
    pub fn score_threshold(&self) -> f64 {
        self.inner.score_threshold
    }
}

/// Tokenize a string with the same rules as the internal `DefaultTokenizer`.
/// Useful for callers that want to pre-process a query (e.g. to feed `autocomplete`
/// with the last token of a multi-word query).
#[wasm_bindgen(js_name = defaultTokenize)]
pub fn default_tokenize(text: &str) -> Vec<JsValue> {
    DefaultTokenizer
        .tokenize(text)
        .into_iter()
        .map(JsValue::from)
        .collect()
}

#[wasm_bindgen(js_name = defaultMaxEditDistance)]
pub fn default_max_edit_distance() -> usize {
    DEFAULT_MAX_EDIT_DISTANCE
}

#[wasm_bindgen(js_name = defaultMaxEditDistanceBoost)]
pub fn default_max_edit_distance_boost() -> f64 {
    DEFAULT_MAX_EDIT_DISTANCE_BOOST
}

#[wasm_bindgen(js_name = defaultMaxPrefixBoost)]
pub fn default_max_prefix_boost() -> f64 {
    DEFAULT_MAX_PREFIX_BOOST
}

#[wasm_bindgen(js_name = defaultScoreThreshold)]
pub fn default_score_threshold() -> f64 {
    DEFAULT_SCORE_THRESHOLD
}
