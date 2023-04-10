use core::fmt;
use fst::{Automaton, IntoStreamer, Streamer};
use fxhash::{FxBuildHasher, FxHashMap};
use num_traits::cast::FromPrimitive;
use std::iter::FromIterator;

pub type Token = String;
pub type Distance = usize;
pub type Score = f64;
type DocId = usize;
type TokenCount = usize;
type TokenOccurrenceCount = usize;
type DocBoost = f64;

mod levenshtein;

use levenshtein::{Levenshtein, LevenshteinState};

// ------ Defaults ------

pub const DEFAULT_MAX_EDIT_DISTANCE: usize = 1;
pub const DEFAULT_MAX_EDIT_DISTANCE_BOOST: f64 = 2.;
pub const DEFAULT_MAX_PREFIX_BOOST: f64 = 1.5;
pub const DEFAULT_SCORE_THRESHOLD: f64 = 0.48;

pub trait Tokenizer: Clone + fmt::Debug {
    #[must_use]
    fn tokenize(&self, text: &str) -> Vec<Token>;
}

/// The default tokenizer.
///
/// - It's used for document indexing and for the search query parsing.
/// - It replaces characters `:` and `,` with a space, removes `-` and splits the words by a whitespace and converts them to lower case.
/// - If you want to use a custom tokenizer, call the method `LocalSearchBuilder::tokenizer`.
///
/// ## Examples
///
/// ```
/// # use localsearch::{Tokenizer, DefaultTokenizer};
/// assert_eq!(
///     DefaultTokenizer.tokenize("Spider-Man: Far from Home"),
///     ["spiderman", "far", "from", "home"]
/// );
/// ```
#[derive(Clone, Debug)]
pub struct DefaultTokenizer;

impl Tokenizer for DefaultTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        text.replace([':', ','].as_ref(), " ")
            .replace('-', "")
            .split_whitespace()
            .map(str::to_lowercase)
            .collect()
    }
}

// ------ LocalSearchBuilder ------

/// Use `LocalSearchBuilder::new(...)` + `build()` to create a new `LocalSearch` instance.
pub struct LocalSearchBuilder<T, TF = DefaultTokenizer> {
    documents: Vec<T>,
    text_extractor: Box<dyn Fn(&T) -> &str>,
    boost_computer: Option<Box<dyn Fn(&T) -> f64>>,
    tokenizer: TF,
    max_edit_distance: Option<Distance>,
    max_edit_distance_boost: Option<f64>,
    max_prefix_boost: Option<f64>,
    score_threshold: Option<f64>,
}

impl<T> LocalSearchBuilder<T, DefaultTokenizer> {
    /// Creates a new `LocalSearchBuilder` instance.
    ///
    /// _Note:_ You should use `LocalSearch::builder` instead.
    pub fn new(documents: Vec<T>, text_extractor: impl Fn(&T) -> &str + 'static) -> Self {
        Self {
            documents,
            text_extractor: Box::new(text_extractor),
            boost_computer: None,
            tokenizer: DefaultTokenizer,
            max_edit_distance: None,
            max_edit_distance_boost: None,
            max_prefix_boost: None,
            score_threshold: None,
        }
    }
}

impl<T, TF> LocalSearchBuilder<T, TF>
where
    TF: Tokenizer,
{
    /// Sets the boost computer. It's a closure that can boost the score for the particular document.
    /// The final document score is multiplied by the computed value returned from the closure.
    ///
    /// It's called for each document during the search.
    ///
    /// The default boost computer is `|_| 1.0`. (The score is multiplied by 1.0 so it isn't affected at all.)
    ///
    /// # Example
    ///
    /// ```
    /// # use localsearch::LocalSearch;
    /// # struct Record { name: String, imdb_rating: f64, popularity: f64 }
    /// # let downloaded_records = vec![Record { name: "superman".to_owned(), imdb_rating: 6., popularity: 25. }];
    /// # let (max_imdb_rating, max_popularity, imdb_rating_weight, popularity_weight) = (10., 100., 1., 1.);
    ///
    /// let boost_computer = move |record: &Record| {
    ///    let imdb_rating_boost = (record.imdb_rating / max_imdb_rating * imdb_rating_weight).exp();
    ///    let popularity_boost = (record.popularity / max_popularity * popularity_weight).exp();
    ///    imdb_rating_boost * popularity_boost
    /// };
    ///
    /// LocalSearch::builder(downloaded_records, |rec| &rec.name)
    ///    .boost_computer(boost_computer)
    ///    .build();
    /// ```
    #[must_use]
    pub fn boost_computer(mut self, boost_computer: impl Fn(&T) -> f64 + 'static) -> Self {
        self.boost_computer = Some(Box::new(boost_computer));
        self
    }

    /// Sets the tokenizer that is used for document indexing and for the search query parsing.
    ///
    /// The default tokenizer is the function `default_tokenizer`.
    ///
    /// # Example
    ///
    /// ```
    /// use localsearch::{LocalSearch, Tokenizer, Token};
    /// # struct Record { name: String }
    /// # let downloaded_records = vec![Record { name: "superman".to_owned() }];
    ///
    /// #[derive(Clone, Debug)]
    /// struct CustomTokenizer;
    ///
    /// impl Tokenizer for CustomTokenizer {
    ///    fn tokenize(&self, text: &str) -> Vec<Token> {
    ///         text.split_whitespace().map(str::to_lowercase).collect()
    ///    }
    /// }
    ///
    /// LocalSearch::builder(downloaded_records, |rec| &rec.name)
    ///     .tokenizer(CustomTokenizer)
    ///     .build();
    /// ```
    #[must_use]
    pub fn tokenizer<TF2>(self, tokenizer: TF2) -> LocalSearchBuilder<T, TF2>
    where
        TF2: Tokenizer,
    {
        LocalSearchBuilder {
            documents: self.documents,
            text_extractor: self.text_extractor,
            boost_computer: self.boost_computer,
            tokenizer,
            max_edit_distance: self.max_edit_distance,
            max_edit_distance_boost: self.max_edit_distance_boost,
            max_prefix_boost: self.max_prefix_boost,
            score_threshold: self.score_threshold,
        }
    }

    /// Sets the maximum edit distance.
    /// _Edit distance_ is an integer value representing how much two selected strings differ.
    /// It determines which document tokens are related to the tokens extracted from the search query.
    /// Document tokens with the higher edit distance then the set threshold (`max_edit_distance`) are ignored.
    ///
    /// The smaller the edit distance value, the higher the score. Zero means that the strings are identical.
    ///
    /// The default value is `DEFAULT_MAX_EDIT_DISTANCE`.
    /// You can set it to `0` or `max_edit_distance_boost` to `0.` to disable fuzzy search.
    ///
    /// # Example
    ///
    /// ```
    /// # use localsearch::LocalSearch;
    /// # struct Record { name: String }
    /// # let downloaded_records = vec![Record { name: "superman".to_owned() }];
    ///
    /// LocalSearch::builder(downloaded_records, |rec| &rec.name)
    ///     .max_edit_distance(2)
    ///     .build();
    /// ```
    #[must_use]
    pub const fn max_edit_distance(mut self, max_edit_distance: Distance) -> Self {
        self.max_edit_distance = Some(max_edit_distance);
        self
    }

    /// Sets the maximum edit distance boost.
    /// See the method `max_edit_distance` for explanation what is _edit distance_.
    ///
    /// The default value is `DEFAULT_MAX_EDIT_DISTANCE_BOOST`.
    /// You can set it to `0.` or `max_edit_distance` to `0` to disable fuzzy search.
    ///
    /// # Example
    /// ```
    /// # use localsearch::LocalSearch;
    /// # struct Record { name: String }
    /// # let downloaded_records = vec![Record { name: "superman".to_owned() }];
    ///
    /// LocalSearch::builder(downloaded_records, |rec| &rec.name)
    ///     .max_edit_distance_boost(10.)
    ///     .build();
    /// ```
    #[must_use]
    pub const fn max_edit_distance_boost(mut self, max_edit_distance_boost: f64) -> Self {
        self.max_edit_distance_boost = Some(max_edit_distance_boost);
        self
    }

    /// Sets the maximum prefix boost.
    ///
    /// The prefix boost depends on the search query token and the related document token.
    /// Let's imagine one of the search query token is "foobar" and the document token is:
    /// - "bar" => "bar" isn't a prefix at all => no boost
    /// - "foo" => "foo" is a prefix and its length is half => the half boost
    /// - "foobar" => "foobar" it's a special kind of prefix - it's identical to the compared token => maximum boost
    ///
    /// The default value is `DEFAULT_MAX_PREFIX_BOOST`. You can set it to `0.` to disable the prefix search.
    ///
    /// # Example
    /// ```
    /// # use localsearch::LocalSearch;
    /// # struct Record { name: String }
    /// # let downloaded_records = vec![Record { name: "superman".to_owned() }];
    ///
    /// LocalSearch::builder(downloaded_records, |rec| &rec.name)
    ///     .max_prefix_boost(10.)
    ///     .build();
    /// ```
    #[must_use]
    pub const fn max_prefix_boost(mut self, max_prefix_boost: f64) -> Self {
        self.max_prefix_boost = Some(max_prefix_boost);
        self
    }

    /// Sets the score threshold. The value has to be in the interval `0.0` - `1.0`, inclusive.
    /// All results that have the score lower than `top_result_score * score_threshold` are filtered out.
    ///
    /// The default value is `DEFAULT_SCORE_THRESHOLD`. You can set it to `0.` to disable filtering.
    ///
    /// # Example
    /// ```
    /// # use localsearch::LocalSearch;
    /// # struct Record { name: String }
    /// # let downloaded_records = vec![Record { name: "superman".to_owned() }];
    ///
    /// LocalSearch::builder(downloaded_records, |rec| &rec.name)
    ///     .score_threshold(0.)
    ///     .build();
    /// ```
    #[must_use]
    pub const fn score_threshold(mut self, score_threshold: f64) -> Self {
        self.score_threshold = Some(score_threshold);
        self
    }

    /// Creates a new `LocalSearch` instance from the `LocalSearchBuilder`.
    ///
    /// _Note:_ Indexing may took awhile for a big dataset.
    #[must_use]
    pub fn build(self) -> LocalSearch<T, TF> {
        // Container for documents with additional data.
        // - `DocId` is a document position from the original document `Vec`.
        // - `T` is the document type.
        // - `TokenCount` is the number of tokens extracted from the document by `text_extractor`. And it's used in the tf-idf algorithm.
        // - `DocBoost` is computed by `boost_computer` and it affects the score for the particular document.
        let mut documents = FxHashMap::<DocId, (T, TokenCount, DocBoost)>::with_capacity_and_hasher(
            self.documents.len(),
            FxBuildHasher::default(),
        );
        let boost_computer = self.boost_computer.unwrap_or_else(|| Box::new(|_| 1.0));

        // `token_and_pairs_map` will be later a part of the index and it contains important data for the tf-idf algorithm.
        let mut token_and_pairs_map =
            FxHashMap::<Token, FxHashMap<DocId, TokenOccurrenceCount>>::with_capacity_and_hasher(
                self.documents.len(),
                FxBuildHasher::default(),
            );

        // These loops fill `documents` and `token_and_pairs_map`.
        for (doc_id, document) in self.documents.into_iter().enumerate() {
            let text = (self.text_extractor)(&document);
            let tokens = self.tokenizer.tokenize(text);
            let token_count = tokens.len();
            let doc_boost = boost_computer(&document);
            documents.insert(doc_id, (document, token_count, doc_boost));

            for token in tokens {
                token_and_pairs_map
                    .entry(token)
                    .and_modify(|doc_id_token_occurrence_count_pairs| {
                        doc_id_token_occurrence_count_pairs
                            .entry(doc_id)
                            .and_modify(|count| *count += 1)
                            .or_insert(1);
                    })
                    .or_insert_with(|| FxHashMap::from_iter(vec![(doc_id, 1)]));
            }
        }

        // We have to convert `HashMap` to `Vec` so we can sort data by key and later split it into two vectors.
        let mut token_and_pairs_vec = token_and_pairs_map.into_iter().collect::<Vec<_>>();

        // `fst::Map` can be constructed only with lexicographically ordered keys.
        token_and_pairs_vec.sort_unstable_by(|(token_a, _), (token_b, _)| token_a.cmp(token_b));

        // We can store only values with type `u64` in `fst::Map`.
        // That's why we have to split our keys and values from `token_and_pairs_vec` into vectors,
        // store only keys/tokens in `fst::Map` as keys and "connect" them by their positions:
        // i.e. A `fst::Map` value will represent the position of the pairs in the vector
        // for the associated `fst::Map` key (aka token).
        let (tokens, pairs): (Vec<_>, Vec<_>) = token_and_pairs_vec.into_iter().unzip();
        let index = Index {
            token_and_pair_index_map: fst::Map::from_iter(tokens.into_iter().zip(0..))
                .expect("build fst map from given documents"),
            pairs,
        };

        LocalSearch {
            documents,
            tokenizer: self.tokenizer,
            max_edit_distance: self.max_edit_distance.unwrap_or(DEFAULT_MAX_EDIT_DISTANCE),
            max_edit_distance_boost: self
                .max_edit_distance_boost
                .unwrap_or(DEFAULT_MAX_EDIT_DISTANCE_BOOST),
            max_prefix_boost: self.max_prefix_boost.unwrap_or(DEFAULT_MAX_PREFIX_BOOST),
            score_threshold: self.score_threshold.unwrap_or(DEFAULT_SCORE_THRESHOLD),
            index,
        }
    }
}

// ------ Index ------

#[derive(Debug, Clone)]
pub struct Index {
    token_and_pair_index_map: fst::Map<Vec<u8>>,
    pairs: Vec<FxHashMap<DocId, TokenOccurrenceCount>>,
}

// ------ RelatedTokenData ------

#[derive(Clone, Copy)]
struct RelatedTokenData {
    distance: Option<Distance>,
    prefix_ratio: Option<f64>,
}

// ------ LocalSearch ------

pub struct LocalSearch<T, TF = DefaultTokenizer> {
    documents: FxHashMap<DocId, (T, TokenCount, DocBoost)>,
    tokenizer: TF,
    pub max_edit_distance: Distance,
    pub max_edit_distance_boost: f64,
    pub max_prefix_boost: f64,
    pub score_threshold: f64,
    index: Index,
}

impl<T, TF> fmt::Debug for LocalSearch<T, TF>
where
    T: fmt::Debug,
    TF: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalSearch")
            .field("documents", &self.documents)
            .field("tokenizer", &"Box<dyn Fn(&str) -> Vec<Token>>")
            .field("max_edit_distance", &self.max_edit_distance)
            .field("max_edit_distance_boost", &self.max_edit_distance_boost)
            .field("max_prefix_boost", &self.max_prefix_boost)
            .field("score_threshold", &self.score_threshold)
            .field("index", &self.index)
            .finish()
    }
}

// impl<T> Clone for LocalSearch<T>
// where
//     T: Clone,
// {
//     fn clone(&self) -> Self {
//         Self {
//             documents: self.documents.clone(),
//             tokenizer: Box::new(self.tokenizer.clone()),
//             max_edit_distance: self.max_edit_distance.clone(),
//             max_edit_distance_boost: self.max_edit_distance_boost.clone(),
//             max_prefix_boost: self.max_prefix_boost.clone(),
//             score_threshold: self.score_threshold.clone(),
//             index: self.index.clone(),
//         }
//     }
// }

impl<T> LocalSearch<T, DefaultTokenizer> {
    // ------ pub ------

    /// Creates a new `LocalSearchBuilder` instance.
    ///
    /// It's the standard way to initialize `LocalSearch`.
    ///
    /// # Arguments
    ///
    /// * `documents`
    ///   - A list of documents you want to index.
    ///   - They will be stored in the `LocalSearch` instance.
    ///   - The search results will be these documents with associated extra data like `score`.
    ///
    /// * `text_extractor`
    ///   - It's a closure that knows how to retrive a text from a document.
    ///   - These extracted text strings represent the content for searching.
    ///
    /// # Example
    ///
    /// ```
    /// # use localsearch::LocalSearch;
    /// # struct Record { name: String, popularity: f64 }
    /// # let downloaded_records = vec![Record { name: "superman".to_owned(), popularity: 8. }];
    ///
    /// LocalSearch::builder(downloaded_records, |rec| &rec.name)
    ///     .boost_computer(|rec| rec.popularity * 0.5)
    ///     .score_threshold(0.48)
    ///     .build();
    /// ```
    pub fn builder(
        documents: Vec<T>,
        text_extractor: impl Fn(&T) -> &str + 'static,
    ) -> LocalSearchBuilder<T> {
        LocalSearchBuilder::new(documents, text_extractor)
    }
}

impl<T, TF> LocalSearch<T, TF>
where
    TF: Tokenizer,
{
    /// Search documents according to the provided search query ordered by their score descending.
    ///
    /// It also filters out any result which doesn't meet the [`Self::score_threshold`] given the
    /// highest score in the search results.
    ///
    /// # Arguments
    ///
    /// * `query`
    ///   - A string that will be tokenized and used for searching.
    ///
    /// * `max_results`
    ///   - The maximum number of results that can be returned.
    ///
    /// # Example
    ///
    /// ```
    /// use localsearch::{LocalSearch, Score};
    ///
    /// #[derive(Clone)]
    /// struct Record { name: String }
    ///
    /// fn search(query: &str, local_search: &LocalSearch<Record>, max_results: usize) -> Vec<(Record, Score)> {
    ///     local_search
    ///         .search(query, max_results)
    ///         .into_iter()
    ///         .map(|(record, score)| (record.clone(), score))
    ///         .collect()
    /// }
    /// ```
    #[must_use]
    pub fn search(&self, query: &str, max_results: usize) -> Vec<(&T, Score)> {
        let mut doc_ids_and_scores = self
            .tokenizer
            .tokenize(query)
            .into_iter()
            // Find related tokens to the tokens parsed from the query.
            // Related tokens are determined by the prefix search or distance search or by the exact match with the query tokens.
            .flat_map(|token| self.related_tokens(&token))
            // Score documents according to the token's `RelatedTokenDat`.
            .flat_map(|(token, data)| self.score_docs(&token, data))
            // Aggregate documents.
            .fold(FxHashMap::default(), |mut results, (doc_id, score)| {
                results
                    .entry(doc_id)
                    .and_modify(|merged_score| *merged_score += score)
                    .or_insert(score);
                results
            })
            .into_iter()
            .collect::<Vec<_>>();

        // Sort by the score.
        doc_ids_and_scores.sort_unstable_by(|(_, score_a), (_, score_b)| {
            score_b.partial_cmp(score_a).expect("sort scores")
        });

        doc_ids_and_scores.truncate(max_results);

        // Filter out results with a too low score. And change `doc_id` for the original document.

        let highest_score = doc_ids_and_scores
            .first()
            .map(|(_, score)| *score)
            .unwrap_or_default();
        let absolute_score_threshold = highest_score * self.score_threshold;

        doc_ids_and_scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                if score <= absolute_score_threshold {
                    return None;
                }
                Some((
                    self.documents
                        .get(&doc_id)
                        .map(|(doc, _, _)| doc)
                        .expect("get document"),
                    score,
                ))
            })
            .collect()
    }

    /// Search for document tokens that have the provided string as a prefix.
    ///
    /// # Arguments
    ///
    /// * `query_token`
    ///   - A string that will be used for searching.
    ///
    /// * `max_results`
    ///   - The maximum number of results that can be returned.
    ///
    /// # Examples
    ///
    /// ```
    /// # use localsearch::LocalSearch;
    /// use localsearch::Tokenizer;
    /// # struct Record { name: String }
    ///
    /// fn autocomplete(query: &str, local_search: &LocalSearch<Record>, max_results: usize) -> Vec<String> {
    ///     if let Some(last_token) = localsearch::DefaultTokenizer.tokenize(query).last() {
    ///         local_search.autocomplete(last_token, max_results)
    ///     } else {
    ///         Vec::new()
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn autocomplete(&self, query_token: &str, max_results: usize) -> Vec<String> {
        let mut token_stream = self
            .index
            .token_and_pair_index_map
            // Search by prefix.
            .search(fst::automaton::Str::new(query_token).starts_with())
            .into_stream();

        let mut tokens = Vec::new();
        // Fill `tokens` from the `fst` byte stream until the `max_result` threshold is reached.
        while let Some((token, _)) = token_stream.next() {
            let token = String::from_utf8(token.to_vec())
                .expect("cannot convert token to valid UTF-8 String");
            tokens.push(token);
            if tokens.len() == max_results {
                break;
            }
        }
        tokens
    }

    // ------ private ------

    // Related tokens are determined by the prefix search or distance search or by the exact match with the query tokens.
    fn related_tokens(&self, query_token: &str) -> FxHashMap<Token, RelatedTokenData> {
        let fuzzy_search_enabled = self.max_edit_distance > 0 && self.max_edit_distance_boost > 0.;
        let prefix_search_enabled = self.max_prefix_boost > 0.;

        let mut related_tokens = FxHashMap::default();

        if fuzzy_search_enabled {
            self.add_tokens_in_distance(query_token, &mut related_tokens);
        }
        if prefix_search_enabled {
            self.add_tokens_with_prefix(query_token, &mut related_tokens);
        }
        // If the fuzzy or prefix search is enabled we don't need to search for the exact matches (aka identical tokens)
        // because both search types automatically compute the highest score for the exact match.
        // However we want to display at least exact matches when both search types are disabled.
        if !fuzzy_search_enabled && !prefix_search_enabled {
            self.add_identical_tokens(query_token, &mut related_tokens)
        }

        related_tokens
    }

    fn add_tokens_in_distance(
        &self,
        query_token: &str,
        related_tokens: &mut FxHashMap<Token, RelatedTokenData>,
    ) {
        // Prepare the query for the `fst` to search by the edit distance (aka fuzzy search).
        let lev_query = Levenshtein::new(query_token, self.max_edit_distance)
            .expect("create Levenshtein automaton");

        let mut token_stream = self
            .index
            .token_and_pair_index_map
            .search_with_state(lev_query)
            .into_stream();

        while let Some((token, _, Some(LevenshteinState { distance, .. }))) = token_stream.next() {
            let token = String::from_utf8(token.to_vec())
                .expect("cannot convert token to valid UTF-8 String");

            // Insert or modify `RelatedTokenData` (set `distance`) in `related_tokens`.
            related_tokens
                .entry(token)
                .and_modify(|related_token_data| related_token_data.distance = distance)
                .or_insert_with(|| RelatedTokenData {
                    distance,
                    prefix_ratio: None,
                });
        }
    }

    fn add_tokens_with_prefix(
        &self,
        query_token: &str,
        related_tokens: &mut FxHashMap<Token, RelatedTokenData>,
    ) {
        let query_token_length =
            f64::from_usize(query_token.len()).expect("query_token_length as f64");
        let mut token_stream = self
            .index
            .token_and_pair_index_map
            // Search by prefix.
            .search(fst::automaton::Str::new(query_token).starts_with())
            .into_stream();

        while let Some((token, _)) = token_stream.next() {
            let token = String::from_utf8(token.to_vec())
                .expect("cannot convert token to valid UTF-8 String");

            let token_length = f64::from_usize(token.len()).expect("token_length as f64");
            let prefix_ratio = Some(query_token_length / token_length);

            // Insert or modify `RelatedTokenData` (set `prefix_ratio`) in `related_tokens`.
            related_tokens
                .entry(token)
                .and_modify(|related_token_data| related_token_data.prefix_ratio = prefix_ratio)
                .or_insert_with(|| RelatedTokenData {
                    distance: None,
                    prefix_ratio,
                });
        }
    }

    fn add_identical_tokens(
        &self,
        query_token: &str,
        related_tokens: &mut FxHashMap<Token, RelatedTokenData>,
    ) {
        let mut token_stream = self
            .index
            .token_and_pair_index_map
            // Search for exact matches.
            .search(fst::automaton::Str::new(query_token))
            .into_stream();

        while let Some((token, _)) = token_stream.next() {
            let token = String::from_utf8(token.to_vec())
                .expect("cannot convert token to valid UTF-8 String");

            related_tokens
                .entry(token)
                .or_insert_with(|| RelatedTokenData {
                    distance: None,
                    prefix_ratio: None,
                });
        }
    }

    fn score_docs(&self, token: &str, token_data: RelatedTokenData) -> FxHashMap<DocId, Score> {
        self.index
            .token_and_pair_index_map
            .get(token)
            .map(|pair_index| {
                let pair_index = usize::from_u64(pair_index).expect("pair_index as usize");
                let doc_id_token_occurrence_count_pairs =
                    self.index.pairs.get(pair_index).expect("get pairs");

                doc_id_token_occurrence_count_pairs.iter().fold(
                    FxHashMap::default(),
                    |mut results, (doc_id, token_occurrence_count)| {
                        let (token_count, doc_boost) = self
                            .documents
                            .get(doc_id)
                            .map(|(_, count, doc_boost)| (*count, *doc_boost))
                            .expect("get token count and document boost");
                        let tf_idf = self.tf_idf(
                            *token_occurrence_count,
                            token_count,
                            doc_id_token_occurrence_count_pairs.len(),
                        );
                        let score = self.score(tf_idf, token_data, doc_boost);
                        results.insert(*doc_id, score);
                        results
                    },
                )
            })
            .unwrap_or_default()
    }

    fn score(&self, tf_idf: f64, token_data: RelatedTokenData, doc_boost: DocBoost) -> Score {
        let distance_boost = token_data.distance.map_or(1., |distance| {
            let distance_difference = f64::from_usize(self.max_edit_distance - distance)
                .expect("distance_difference as f64");
            distance_difference.mul_add(self.max_edit_distance_boost, 1.)
        });

        let prefix_boost = token_data
            .prefix_ratio
            .map_or(1., |ratio| ratio.mul_add(self.max_prefix_boost, 1.));

        (1. + tf_idf) * distance_boost * prefix_boost * doc_boost
    }

    // https://towardsdatascience.com/tf-term-frequency-idf-inverse-document-frequency-from-scratch-in-python-6c2b61b78558
    fn tf_idf(
        &self,
        token_occurrence_count: usize,
        token_count: usize,
        num_of_docs_with_token: usize,
    ) -> f64 {
        let token_occurrence_count =
            f64::from_usize(token_occurrence_count).expect("token_occurrence_count as f64");
        let token_count = f64::from_usize(token_count).expect("token_count as f64");
        let num_of_docs_with_token =
            f64::from_usize(num_of_docs_with_token).expect("num_of_docs_with_token as f64");
        let document_count =
            f64::from_usize(self.documents.len()).expect("self.documents.len() as f64");

        // Term Frequency (TF)
        // tf(t,d) = count of t in d / number of words in d
        let tf = token_occurrence_count / token_count;

        // Document Frequency (DF)
        // df(t) = occurrence of t in documents
        let df = num_of_docs_with_token;

        // Inverse Document Frequency (IDF)
        // idf(t) = log(N/df)
        //
        // Note: Author of the article linked above suggests to add 1 (df + 1).
        // However it would make the algorithm less accurate and we compute tf-idf only for documents
        // that contain the given term - it means our df >= 1.
        let n = document_count;
        let idf = (n / df).log10();

        // tf-idf(t, d) = tf(t, d) * log(N/(df + 1))
        tf * idf
    }
}

#[cfg(test)]
mod tests {
    use super::{DocId, LocalSearch};
    use num_traits::cast::FromPrimitive;

    #[test]
    fn tf_idf_test() {
        // Example data and computed values taken from https://en.wikipedia.org/wiki/Tf%E2%80%93idf

        // ---- ARRANGE ----
        let records = vec![
            "this is a a sample",
            "this is another another example example example",
        ];

        let local_search = LocalSearch::builder(records, |record| record).build();

        let round_f64 = |number: f64, places: u32| {
            let factor = f64::from(10_u32.pow(places));
            (number * factor).round() / factor
        };

        // The closure body is inspired by the function `LocalSearch::score_docs`.
        let compute_tf_idfs = |token: &str| -> Vec<(DocId, f64)> {
            local_search
                .index
                .token_and_pair_index_map
                .get(&token)
                .map(|pair_index| {
                    let pair_index = usize::from_u64(pair_index).expect("pair_index as usize");
                    let doc_id_token_occurrence_count_pairs =
                        local_search.index.pairs.get(pair_index).expect("get pairs");

                    doc_id_token_occurrence_count_pairs
                        .iter()
                        .map(|(doc_id, token_occurrence_count)| {
                            let (token_count, _doc_boost) = local_search
                                .documents
                                .get(doc_id)
                                .map(|(_, count, doc_boost)| (*count, *doc_boost))
                                .expect("get token count and document boost");
                            let tf_idf = local_search.tf_idf(
                                *token_occurrence_count,
                                token_count,
                                doc_id_token_occurrence_count_pairs.len(),
                            );
                            let rounded_tf_idf = round_f64(tf_idf, 3);
                            (*doc_id, rounded_tf_idf)
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        // ---- ACT ----
        let tf_idf_for_this = compute_tf_idfs("this");
        let tf_idf_for_example = compute_tf_idfs("example");

        // ---- ASSERT ----
        let expected_tf_idf_for_this = [(0, 0.0), (1, 0.0)];

        // Note: tf-idf isn't computed for the document with the `doc_id` 0
        // because that document doesn't contain the term "example".
        let expected_tf_idf_for_example = [(1, 0.129)];

        assert_eq!(tf_idf_for_this, expected_tf_idf_for_this);
        assert_eq!(tf_idf_for_example, expected_tf_idf_for_example);
    }

    #[test]
    fn autocomplete_test() {
        // ---- ARRANGE ----
        let records = vec![
            "apple",
            "this is a a sample",
            "this is another another example example example",
            "orange",
        ];

        let local_search = LocalSearch::builder(records, |record| record).build();

        // ---- ACT ----
        let results = local_search.autocomplete("a", 5);

        // ---- ASSERT ----
        assert_eq!(results, ["a", "another", "apple"]);
    }

    #[test]
    fn search_test() {
        // ---- ARRANGE ----
        let records = [
            "apple",
            "this is a a sample",
            "this is another another example example example",
            "another document",
            "orange",
        ]
        .iter()
        .enumerate()
        .collect();

        let local_search = LocalSearch::builder(records, |(_, text)| text).build();

        // ---- ACT ----
        let results = local_search.search("thi range", 5);
        let doc_id_results = results
            .into_iter()
            .map(|((doc_id, _), _)| *doc_id)
            .collect::<Vec<_>>();

        // ---- ASSERT ----
        // - doc_id 1: `thi` matches `this`
        // - doc_id 2: `thi` matches `this`
        // - doc_id 4: `range` matches `orange`
        let expected_results = [1, 2, 4];
        assert_eq!(doc_id_results, expected_results);
    }
}
