#![allow(
    clippy::wildcard_imports,
    clippy::used_underscore_binding,
    clippy::future_not_send
)]

use seed::{prelude::*, *};

mod localsearch_search_panel;

// ------ ------
//     Init
// ------ ------

fn init(_: Url, _: &mut impl Orders<Msg>) -> Model {
    Model {
        cinemeta_localsearch_20_000: localsearch_search_panel::init(
            "Cinemeta (localsearch-20_000)",
            "/data/cinemeta_20_000.json",
        ),
    }
}

// ------ ------
//     Model
// ------ ------

struct Model {
    cinemeta_localsearch_20_000: localsearch_search_panel::Model,
}

// ------ ------
//    Update
// ------ ------

enum Msg {
    CinemetaLocalsearch20000(localsearch_search_panel::Msg),
}

fn update(msg: Msg, model: &mut Model, orders: &mut impl Orders<Msg>) {
    match msg {
        Msg::CinemetaLocalsearch20000(msg) => localsearch_search_panel::update(
            msg,
            &mut model.cinemeta_localsearch_20_000,
            &mut orders.proxy(Msg::CinemetaLocalsearch20000),
        ),
    }
}

// ------ ------
//     View
// ------ ------

fn view(model: &Model) -> Node<Msg> {
    div![
        style! {
            St::Display => "flex",
            St::FlexWrap => "wrap",
            St::MaxWidth => vw(100),
            St::MaxHeight => vh(100),
            St::Overflow => "auto",
        },
        localsearch_search_panel::view(&model.cinemeta_localsearch_20_000)
            .map_msg(Msg::CinemetaLocalsearch20000),
    ]
}

// ------ ------
//     Start
// ------ ------

// (This function is invoked by `init` function in `index.html`.)
#[wasm_bindgen(start)]
pub fn start() {
    // Mount the `app` to the element with the `id` "app".
    App::start("app", init, update, view);
}
