//! Face geometry + particle generation.
//!
//! Ported from `~/src/avatar/static/viz/face-gen.js`. A face is built
//! by sampling a procedurally-shaded depth map (`depth_at`) weighted by
//! a presence mask (`mask_at`), then colouring each sampled particle
//! by depth + per-feature accents (`color_point`).

pub mod generate;
pub mod geometry;

pub use generate::{generate_face, Particle};
