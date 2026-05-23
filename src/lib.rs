//! ghostly — a Rust port of the avatar particle-face visuals.
//!
//! `~/src/avatar` is a Python keynote-avatar app whose striking visual
//! identity is a 60-80K particle face renderer that runs in the
//! browser (Canvas 2D, driven by JS modules in `static/viz/`). This
//! crate ports those visuals to Rust so they can render server-side
//! and feed an H.264 video stream — the same shape Eliza's tile uses
//! to broadcast over MoQ today.
//!
//! ## Status
//!
//! - **Oblivion** is fully ported — geometry (with horns), palette,
//!   contour set, "ominous sink" transition, voronoi-mesh overlay.
//! - **Narrator**, **Utopia**, **Eliza** are placeholders — they
//!   build a clean character record with the right palette / geometry
//!   roughly sketched, so the registry + CLI + render pipeline can be
//!   exercised. Each module's doc comment names the specific work
//!   left.
//!
//! ## Quick start
//!
//! ```no_run
//! use ghostly::{characters, render::{Renderer, RenderSettings, FaceState}};
//!
//! let character = characters::by_name("oblivion").unwrap();
//! let mut state = FaceState::new(&character, 12_000, 2.8, 42);
//! let mut renderer = Renderer::new(RenderSettings::default()).unwrap();
//! let pixmap = renderer.render(&character, &state, 0.0);
//! pixmap.save_png("oblivion.png").unwrap();
//! ```

pub mod character;
pub mod characters;
pub mod face;
pub mod render;

pub use character::{
    Character, Contour, ContourBaseline, Displace, Geometry, Palette, RenderConfig, Transition,
    VoronoiMesh,
};
pub use face::{generate_face, Particle};
pub use render::{FaceState, RenderSettings, Renderer};
