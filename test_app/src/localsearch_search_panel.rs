use localsearch::{self, LocalSearch, Score};
use seed::{fetch, prelude::*, *};
use serde::Deserialize;
use std::str::FromStr;
use web_sys::Performance;

// ------ ------
//     Model
// ------ ------

pub struct Model {
    title: &'static str,
    download_url: &'static str,
    downloaded_records: Vec<Record>,
    local_search: LocalSearch<Record>,
    download_start: f64,
    download_time: Option<f64>,
    index_time: Option<f64>,
    query: String,
    search_time: Option<f64>,
    max_autocomplete_results: usize,
    score_threshold: f64,
    max_results: usize,
    autocomplete_results: Vec<String>,
    results: Vec<(Record, Score)>,
    performance: Performance,
    index_options: IndexOptions,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Record {
    id: String,
    name: String,
    #[serde(rename(deserialize = "imdbRating"))]
    imdb_rating: f64,
    popularity: f64,
}

#[derive(Copy, Clone)]
struct IndexOptions {
    imdb_rating_weight: f64,
    popularity_weight: f64,
}

// ------ ------
//     Init
// ------ ------

pub fn init(title: &'static str, download_url: &'static str) -> Model {
    Model {
        title,
        download_url,
        downloaded_records: Vec::new(),
        local_search: LocalSearch::builder(Vec::new(), |rec: &Record| &rec.name).build(),
        download_start: 0.,
        download_time: None,
        index_time: None,
        query: "".to_owned(),
        search_time: None,
        max_autocomplete_results: 5,
        score_threshold: 0.48,
        max_results: 15,
        autocomplete_results: Vec::new(),
        results: Vec::new(),
        performance: window().performance().expect("get `Performance`"),
        index_options: IndexOptions {
            imdb_rating_weight: 1.0,
            popularity_weight: 1.0,
        },
    }
}

// ------ ------
//    Update
// ------ ------

pub enum Msg {
    Download,
    Downloaded(fetch::Result<Vec<Record>>),
    ImdbRatingWeightChanged(String),
    PopularityWeightChanged(String),
    Index,
    MaxAutocompleteResultsChanged(String),
    ScoreThresholdChanged(String),
    MaxResultsChanged(String),
    QueryChanged(String),
    Search,
}

async fn fetch_records(url: &'static str) -> Msg {
    Msg::Downloaded(async { fetch(url).await?.json().await }.await)
}

fn index(
    downloaded_records: Vec<Record>,
    index_options: IndexOptions,
    score_threshold: f64,
) -> LocalSearch<Record> {
    let max_imdb_rating = 10.;

    let max_popularity = downloaded_records
        .iter()
        .map(|record| record.popularity)
        .max_by(|popularity_a, popularity_b| popularity_a.partial_cmp(popularity_b).unwrap())
        .unwrap_or_default();

    let IndexOptions {
        imdb_rating_weight,
        popularity_weight,
    } = index_options;

    let boost_computer = move |record: &Record| {
        let imdb_rating_boost = (record.imdb_rating / max_imdb_rating * imdb_rating_weight).exp();
        let popularity_boost = (record.popularity / max_popularity * popularity_weight).exp();
        imdb_rating_boost * popularity_boost
    };

    LocalSearch::builder(downloaded_records, |rec| &rec.name)
        .boost_computer(boost_computer)
        .score_threshold(score_threshold)
        .build()
}

fn search(
    query: &str,
    local_search: &LocalSearch<Record>,
    max_results: usize,
) -> Vec<(Record, Score)> {
    local_search
        .search(query, max_results)
        .into_iter()
        .map(|(record, score)| (record.clone(), score))
        .collect()
}

fn autocomplete(
    query: &str,
    local_search: &LocalSearch<Record>,
    max_results: usize,
) -> Vec<String> {
    localsearch::default_tokenizer(query)
        .last()
        .map_or(Vec::new(), |last_token| {
            local_search.autocomplete(last_token, max_results)
        })
}

pub fn update(msg: Msg, model: &mut Model, orders: &mut impl Orders<Msg>) {
    match msg {
        Msg::Download => {
            model.download_start = model.performance.now();
            model.download_time = None;
            orders.perform_cmd(fetch_records(model.download_url));
        }
        Msg::Downloaded(Ok(records)) => {
            model.downloaded_records = records;
            model.download_time = Some(model.performance.now() - model.download_start);
        }
        Msg::Downloaded(Err(err)) => {
            log!("Download error", err);
        }
        Msg::ImdbRatingWeightChanged(imdb_rating_weight) => {
            if let Ok(imdb_rating_weight) = f64::from_str(&imdb_rating_weight) {
                model.index_options.imdb_rating_weight = imdb_rating_weight;
            } else {
                orders.skip();
            }
        }
        Msg::PopularityWeightChanged(popularity_weight) => {
            if let Ok(popularity_weight) = f64::from_str(&popularity_weight) {
                model.index_options.popularity_weight = popularity_weight;
            } else {
                orders.skip();
            }
        }
        Msg::Index => {
            let records = model.downloaded_records.clone();
            let index_start = model.performance.now();
            model.local_search = index(records, model.index_options, model.score_threshold);
            model.index_time = Some(model.performance.now() - index_start);
            orders.send_msg(Msg::Search);
        }
        Msg::MaxAutocompleteResultsChanged(max_results) => {
            if let Ok(max_results) = usize::from_str(&max_results) {
                model.max_autocomplete_results = max_results;
                orders.send_msg(Msg::Search);
            } else {
                orders.skip();
            }
        }
        Msg::ScoreThresholdChanged(score_threshold_percent) => {
            if let Ok(score_threshold_percent) = f64::from_str(&score_threshold_percent) {
                let score_threshold = score_threshold_percent / 100.;
                model.score_threshold = score_threshold;
                model.local_search.score_threshold = score_threshold;
                orders.send_msg(Msg::Search);
            } else {
                orders.skip();
            }
        }
        Msg::MaxResultsChanged(max_results) => {
            if let Ok(max_results) = usize::from_str(&max_results) {
                model.max_results = max_results;
                orders.send_msg(Msg::Search);
            } else {
                orders.skip();
            }
        }
        Msg::QueryChanged(query) => {
            model.query = query;
            orders.send_msg(Msg::Search);
        }
        Msg::Search => {
            let search_start = model.performance.now();
            model.results = search(&model.query, &model.local_search, model.max_results);
            model.search_time = Some(model.performance.now() - search_start);
            model.autocomplete_results = autocomplete(
                &model.query,
                &model.local_search,
                model.max_autocomplete_results,
            );
        }
    }
}

// ------ ------
//     View
// ------ ------

pub fn view(model: &Model) -> Node<Msg> {
    div![
        style! {
            St::Padding => px(10),
            St::MinWidth => px(320),
        },
        h2![model.title,],
        view_download(model),
        view_imdb_rating_weight(model),
        view_popularity_weight(model),
        view_index(model),
        view_max_autocomplete_results(model),
        view_score_threshold(model),
        view_max_results(model),
        view_query(model),
        view_autocomplete_results(model),
        view_results(model),
    ]
}

pub fn view_download(model: &Model) -> Node<Msg> {
    div![
        style! {
            St::Display => "flex",
            St::AlignItems => "center",
            St::Padding => "10px 0",
        },
        div![
            style! {
                St::Cursor => "pointer",
                St::Padding => "5px 15px",
                St::BackgroundColor => "lightgreen",
                St::BorderRadius => px(10),
            },
            ev(Ev::Click, |_| Msg::Download),
            "Download & Deserialize"
        ],
        div![
            style! {
                St::Padding => "0 10px",
            },
            format!(
                "{} ms",
                model
                    .download_time
                    .as_ref()
                    .map_or("-".to_owned(), ToString::to_string)
            ),
        ],
    ]
}

pub fn view_imdb_rating_weight(model: &Model) -> Node<Msg> {
    div![
        style! {
            St::Display => "flex",
            St::AlignItems => "center",
            St::Padding => "10px 0",
        },
        div!["IMDB rating weight:"],
        input![
            style! {
                St::Padding => "3px 8px",
                St::Margin => "0 10px",
                St::Border => "2px solid black",
            },
            attrs! {
                At::Value => model.index_options.imdb_rating_weight,
                At::Type => "number",
                At::Step => 0.01,
            },
            input_ev(Ev::Input, Msg::ImdbRatingWeightChanged),
        ],
        "(click \"Index\" to apply changes)",
    ]
}

pub fn view_popularity_weight(model: &Model) -> Node<Msg> {
    div![
        style! {
            St::Display => "flex",
            St::AlignItems => "center",
            St::Padding => "10px 0",
        },
        div!["Popularity weight:"],
        input![
            style! {
                St::Padding => "3px 8px",
                St::Margin => "0 10px",
                St::Border => "2px solid black",
            },
            attrs! {
                At::Value => model.index_options.popularity_weight,
                At::Type => "number",
                At::Step => 0.01,
            },
            input_ev(Ev::Input, Msg::PopularityWeightChanged),
        ],
        "(click \"Index\" to apply changes)",
    ]
}

pub fn view_index(model: &Model) -> Node<Msg> {
    div![
        style! {
            St::Display => "flex",
            St::AlignItems => "center",
            St::Padding => "10px 0",
        },
        div![
            style! {
                St::Cursor => "pointer",
                St::Padding => "5px 15px",
                St::BackgroundColor => "lightblue",
                St::BorderRadius => px(10),
            },
            ev(Ev::Click, |_| Msg::Index),
            "Index"
        ],
        div![
            style! {
                St::Padding => "0 10px",
            },
            format!(
                "{} ms",
                model
                    .index_time
                    .as_ref()
                    .map_or("-".to_owned(), ToString::to_string)
            ),
        ],
    ]
}

pub fn view_max_autocomplete_results(model: &Model) -> Node<Msg> {
    div![
        style! {
            St::Display => "flex",
            St::AlignItems => "center",
            St::Padding => "10px 0",
        },
        div!["Max autocomplete results:"],
        input![
            style! {
                St::Padding => "3px 8px",
                St::Margin => "0 10px",
                St::Border => "2px solid black",
            },
            attrs! {
                At::Value => model.max_autocomplete_results,
                At::Type => "number",
            },
            input_ev(Ev::Input, Msg::MaxAutocompleteResultsChanged),
        ],
    ]
}

pub fn view_score_threshold(model: &Model) -> Node<Msg> {
    div![
        style! {
            St::Display => "flex",
            St::AlignItems => "center",
            St::Padding => "10px 0",
        },
        div!["Score threshold (%):"],
        input![
            style! {
                St::Padding => "3px 8px",
                St::Margin => "0 10px",
                St::Border => "2px solid black",
            },
            attrs! {
                At::Value => model.score_threshold * 100.,
                At::Type => "number",
                At::Step => 0.01,
            },
            input_ev(Ev::Input, Msg::ScoreThresholdChanged),
        ],
    ]
}

pub fn view_max_results(model: &Model) -> Node<Msg> {
    div![
        style! {
            St::Display => "flex",
            St::AlignItems => "center",
            St::Padding => "10px 0",
        },
        div!["Max results:"],
        input![
            style! {
                St::Padding => "3px 8px",
                St::Margin => "0 10px",
                St::Border => "2px solid black",
            },
            attrs! {
                At::Value => model.max_results,
                At::Type => "number",
            },
            input_ev(Ev::Input, Msg::MaxResultsChanged),
        ],
    ]
}

pub fn view_query(model: &Model) -> Node<Msg> {
    div![
        style! {
            St::Display => "flex",
            St::AlignItems => "center",
            St::Padding => "10px 0",
        },
        div!["Query:"],
        input![
            style! {
                St::Padding => "3px 8px",
                St::Margin => "0 10px",
                St::Border => "2px solid black",
            },
            attrs! {
                At::Value => model.query,
            },
            input_ev(Ev::Input, Msg::QueryChanged),
        ],
        div![format!(
            "{} ms",
            model
                .search_time
                .as_ref()
                .map_or("-".to_owned(), ToString::to_string)
        ),],
    ]
}

pub fn view_autocomplete_results(model: &Model) -> Node<Msg> {
    div![model.autocomplete_results.join(" - ")]
}

pub fn view_results(model: &Model) -> Node<Msg> {
    table![
        style! {
            St::Padding => "10px 0",
        },
        thead![tr![
            th!["Id"],
            th!["Name"],
            th!["Score"],
            th!["IMDB"],
            th!["Popularity"]
        ]],
        tbody![model.results.iter().enumerate().map(view_result)]
    ]
}

pub fn view_result(result_item_data: (usize, &(Record, Score))) -> Node<Msg> {
    let (index, (record, score)) = result_item_data;
    tr![
        style! {
            St::BackgroundColor => if index % 2 == 0 { Some("aliceblue") } else { None },
        },
        td![
            style! {
                St::Padding => px(10),
            },
            &record.id,
        ],
        td![
            style! {
                St::Padding => px(10),
            },
            &record.name,
        ],
        td![
            style! {
                St::Padding => px(10),
            },
            score,
        ],
        td![
            style! {
                St::Padding => px(10),
            },
            &record.imdb_rating,
        ],
        td![
            style! {
                St::Padding => px(10),
            },
            &record.popularity,
        ],
    ]
}
