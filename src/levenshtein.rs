use std::cmp;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt;

// @TODO remove this file and `utf8-ranges` from dependencies
// once https://github.com/BurntSushi/fst/pull/99 is merged and released.

use utf8_ranges::{Utf8Range, Utf8Sequences};

use fst::automaton::Automaton;

const STATE_LIMIT: usize = 10_000; // currently at least 20MB >_<

/// An error that occurred while building a Levenshtein automaton.
///
/// This error is only defined when the `levenshtein` crate feature is enabled.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
pub enum LevenshteinError {
    /// If construction of the automaton reaches some hard-coded limit
    /// on the number of states, then this error is returned.
    ///
    /// The number given is the limit that was exceeded.
    TooManyStates(usize),
}

impl fmt::Display for LevenshteinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TooManyStates(size_limit) => write!(
                f,
                "Levenshtein automaton exceeds size limit of \
                           {} states",
                size_limit
            ),
        }
    }
}

impl std::error::Error for LevenshteinError {}

#[allow(clippy::needless_doctest_main)]
/// A Unicode aware Levenshtein automaton for running efficient fuzzy queries.
///
/// This is only defined when the `levenshtein` crate feature is enabled.
///
/// A Levenshtein automata is one way to search any finite state transducer
/// for keys that *approximately* match a given query. A Levenshtein automaton
/// approximates this by returning all keys within a certain edit distance of
/// the query. The edit distance is defined by the number of insertions,
/// deletions and substitutions required to turn the query into the key.
/// Insertions, deletions and substitutions are based on
/// **Unicode characters** (where each character is a single Unicode scalar
/// value).
///
/// # Example
///
/// This example shows how to find all keys within an edit distance of `1`
/// from `foo`.
///
/// ```ignore
/// use fst::automaton::Levenshtein;
/// use fst::{IntoStreamer, Streamer, Set};
///
/// fn main() {
///     let keys = vec!["fa", "fo", "fob", "focus", "foo", "food", "foul"];
///     let set = Set::from_iter(keys).unwrap();
///
///     let lev = Levenshtein::new("foo", 1).unwrap();
///     let mut stream = set.search(&lev).into_stream();
///
///     let mut keys = vec![];
///     while let Some(key) = stream.next() {
///         keys.push(key.to_vec());
///     }
///     assert_eq!(keys, vec![
///         "fo".as_bytes(),   // 1 deletion
///         "fob".as_bytes(),  // 1 substitution
///         "foo".as_bytes(),  // 0 insertions/deletions/substitutions
///         "food".as_bytes(), // 1 insertion
///     ]);
/// }
/// ```
///
/// This example only uses ASCII characters, but it will work equally well
/// on Unicode characters.
///
/// # Warning: experimental
///
/// While executing this Levenshtein automaton against a finite state
/// transducer will be very fast, *constructing* an automaton may not be.
/// Namely, this implementation is a proof of concept. While I believe the
/// algorithmic complexity is not exponential, the implementation is not speedy
/// and it can use enormous amounts of memory (tens of MB before a hard-coded
/// limit will cause an error to be returned).
///
/// This is important functionality, so one should count on this implementation
/// being vastly improved in the future.
pub struct Levenshtein {
    prog: DynamicLevenshtein,
    dfa: Dfa,
}

impl Levenshtein {
    /// Create a new Levenshtein query.
    ///
    /// The query finds all matching terms that are at most `distance`
    /// edit operations from `query`. (An edit operation may be an insertion,
    /// a deletion or a substitution.)
    ///
    /// If the underlying automaton becomes too big, then an error is returned.
    ///
    /// A `Levenshtein` value satisfies the `Automaton` trait, which means it
    /// can be used with the `search` method of any finite state transducer.
    #[inline]
    pub fn new(query: &str, distance: usize) -> Result<Self, LevenshteinError> {
        let lev = DynamicLevenshtein {
            query: query.to_owned(),
            dist: distance,
        };
        let dfa = DfaBuilder::new(lev.clone()).build()?;
        Ok(Self { prog: lev, dfa })
    }
}

impl fmt::Debug for Levenshtein {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Levenshtein(query: {:?}, distance: {:?})",
            self.prog.query, self.prog.dist
        )
    }
}

#[derive(Clone)]
struct DynamicLevenshtein {
    query: String,
    dist: usize,
}

impl DynamicLevenshtein {
    fn start(&self) -> Vec<usize> {
        (0..=self.query.chars().count()).collect()
    }

    fn is_match(&self, state: &[usize]) -> bool {
        state.last().map_or(false, |&n| n <= self.dist)
    }

    fn can_match(&self, state: &[usize]) -> bool {
        state.iter().min().map_or(false, |&n| n <= self.dist)
    }

    fn accept(&self, state: &[usize], chr: Option<char>) -> Vec<usize> {
        let mut next = vec![state[0] + 1];
        for (i, c) in self.query.chars().enumerate() {
            let cost = if Some(c) == chr { 0 } else { 1 };
            let v = cmp::min(cmp::min(next[i] + 1, state[i + 1] + 1), state[i] + cost);
            next.push(cmp::min(v, self.dist + 1));
        }
        next
    }
}

/// Levenshtein automaton state.
///
/// It is useful for obtaining edit distance while searching.
/// See examples in documentation for `Map::search_with_state`
/// or `Set::search_with_state`.
///
/// This is only defined when the `levenshtein` crate feature is enabled.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct LevenshteinState {
    /// Internal state index.
    pub state_idx: usize,
    /// Levenshtein edit distance.
    pub distance: Option<usize>,
}

impl Automaton for Levenshtein {
    type State = Option<LevenshteinState>;

    #[inline]
    fn start(&self) -> Option<LevenshteinState> {
        Some(LevenshteinState {
            state_idx: 0,
            distance: self.dfa.states[0].distance,
        })
    }

    #[inline]
    fn is_match(&self, state: &Option<LevenshteinState>) -> bool {
        state
            .map(|state| self.dfa.states[state.state_idx].is_match)
            .unwrap_or(false)
    }

    #[inline]
    fn can_match(&self, state: &Option<LevenshteinState>) -> bool {
        state.is_some()
    }

    #[inline]
    fn accept(&self, state: &Option<LevenshteinState>, byte: u8) -> Option<LevenshteinState> {
        state.and_then(|state| {
            self.dfa.states[state.state_idx].next[byte as usize].map(|next_state_idx| {
                LevenshteinState {
                    state_idx: next_state_idx,
                    distance: self.dfa.states[next_state_idx].distance,
                }
            })
        })
    }
}

#[derive(Debug)]
struct Dfa {
    states: Vec<State>,
}

struct State {
    next: [Option<usize>; 256],
    is_match: bool,
    distance: Option<usize>,
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "State {{")?;
        writeln!(f, "  is_match: {:?}", self.is_match)?;
        writeln!(f, "  distance: {:?}", self.distance)?;
        for i in 0..256 {
            if let Some(si) = self.next[i] {
                writeln!(f, "  {:?}: {:?}", i, si)?;
            }
        }
        write!(f, "}}")
    }
}

struct DfaBuilder {
    dfa: Dfa,
    lev: DynamicLevenshtein,
    cache: HashMap<Vec<usize>, usize>,
}

impl DfaBuilder {
    fn new(lev: DynamicLevenshtein) -> Self {
        Self {
            dfa: Dfa {
                states: Vec::with_capacity(16),
            },
            lev,
            cache: HashMap::with_capacity(1024),
        }
    }

    fn build(mut self) -> Result<Dfa, LevenshteinError> {
        let mut stack = vec![self.lev.start()];
        let mut seen = HashSet::new();
        let query = self.lev.query.clone(); // temp work around of borrowck
        while let Some(lev_state) = stack.pop() {
            let dfa_si = self.cached_state(&lev_state).unwrap();
            let mismatch = self.add_mismatch_utf8_states(dfa_si, &lev_state);
            if let Some((next_si, lev_next)) = mismatch {
                if !seen.contains(&next_si) {
                    seen.insert(next_si);
                    stack.push(lev_next);
                }
            }
            for (i, c) in query.chars().enumerate() {
                if lev_state[i] > self.lev.dist {
                    continue;
                }
                let lev_next = self.lev.accept(&lev_state, Some(c));
                let next_si = self.cached_state(&lev_next);
                if let Some(next_si) = next_si {
                    self.add_utf8_sequences(true, dfa_si, next_si, c, c);
                    if !seen.contains(&next_si) {
                        seen.insert(next_si);
                        stack.push(lev_next);
                    }
                }
            }
            if self.dfa.states.len() > STATE_LIMIT {
                return Err(LevenshteinError::TooManyStates(STATE_LIMIT));
            }
        }
        Ok(self.dfa)
    }

    fn cached_state(&mut self, lev_state: &[usize]) -> Option<usize> {
        self.cached(lev_state).map(|(si, _)| si)
    }

    fn cached(&mut self, lev_state: &[usize]) -> Option<(usize, bool)> {
        if !self.lev.can_match(lev_state) {
            return None;
        }
        Some(match self.cache.entry(lev_state.to_vec()) {
            Entry::Occupied(v) => (*v.get(), true),
            Entry::Vacant(v) => {
                let is_match = self.lev.is_match(lev_state);
                self.dfa.states.push(State {
                    next: [None; 256],
                    is_match,
                    distance: lev_state.last().copied(),
                });
                (*v.insert(self.dfa.states.len() - 1), false)
            }
        })
    }

    fn add_mismatch_utf8_states(
        &mut self,
        from_si: usize,
        lev_state: &[usize],
    ) -> Option<(usize, Vec<usize>)> {
        let mismatch_state = self.lev.accept(lev_state, None);
        let to_si = match self.cached(&mismatch_state) {
            None => return None,
            Some((si, _)) => si,
            // Some((si, true)) => return Some((si, mismatch_state)),
            // Some((si, false)) => si,
        };
        self.add_utf8_sequences(false, from_si, to_si, '\u{0}', '\u{10FFFF}');
        Some((to_si, mismatch_state))
    }

    #[allow(clippy::similar_names)]
    fn add_utf8_sequences(
        &mut self,
        overwrite: bool,
        from_si: usize,
        to_si: usize,
        from_chr: char,
        to_chr: char,
    ) {
        for seq in Utf8Sequences::new(from_chr, to_chr) {
            let mut fsi = from_si;
            for range in &seq.as_slice()[0..seq.len() - 1] {
                let tsi = self.new_state(false);
                self.add_utf8_range(overwrite, fsi, tsi, *range);
                fsi = tsi;
            }
            self.add_utf8_range(overwrite, fsi, to_si, seq.as_slice()[seq.len() - 1]);
        }
    }

    fn add_utf8_range(&mut self, overwrite: bool, from: usize, to: usize, range: Utf8Range) {
        for b in range.start as usize..=range.end as usize {
            if overwrite || self.dfa.states[from].next[b].is_none() {
                self.dfa.states[from].next[b] = Some(to);
            }
        }
    }

    fn new_state(&mut self, is_match: bool) -> usize {
        self.dfa.states.push(State {
            next: [None; 256],
            is_match,
            distance: None,
        });
        self.dfa.states.len() - 1
    }
}
