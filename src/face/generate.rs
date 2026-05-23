//! Particle generation — sample N particles from the procedural face.
//!
//! Two-stage process (matches `face-gen.js`):
//!   1. Bake a low-res depth + mask grid (`MAP_W × MAP_H`).
//!   2. Sample `count` particles, weighted by mask. About 92% land on
//!      the face; the rest scatter as ambient stars in the surrounding
//!      space, so the field never has a hard cut-off.

use crate::character::{Geometry, Palette};
use crate::face::geometry::{color_point, depth_at, mask_at};

/// A single particle: target rest position (in normalized face space)
/// + the colour to draw it in.
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    /// Rest position in particle-space (scaled, ready to project to
    /// screen — see [`crate::render::Renderer`]).
    pub target: [f32; 3],
    /// Linear-RGB + alpha, all `0..=1`.
    pub color: [f32; 4],
    /// Per-particle seed in `0..1` — drives the displace closure +
    /// alpha shimmer + size flicker.
    pub seed: f32,
    /// True when this particle sits in an eye region — the renderer
    /// gives these a per-frame fire flicker (fast pulse + brightness
    /// boost). Without this the eyes' colour is baked at generation
    /// time and reads static; with this they visibly *crackle*.
    pub is_eye: bool,
}

/// Build `count` particles for a face with the given geometry +
/// palette. Deterministic given `seed` — same call twice yields the
/// same field.
pub fn generate_face(count: usize, scale: f32, geo: &Geometry, pal: &Palette, seed: u64) -> Vec<Particle> {
    let mut rng = fastrand::Rng::with_seed(seed);

    // The depth/mask grid spans a tight box. With horns we extend up
    // and out so the curling ridges aren't clipped.
    const MAP_W: usize = 128;
    let map_h = if geo.horns { 200 } else { 160 };
    let y_lo: f32 = -0.65;
    let y_hi: f32 = if geo.horns { 1.1 } else { 0.85 };
    let y_range = y_hi - y_lo;
    let x_lo: f32 = if geo.horns { -0.65 } else { -0.5 };
    let x_hi: f32 = if geo.horns { 0.65 } else { 0.5 };
    let x_range = x_hi - x_lo;

    let mut depth_map = vec![0.0_f32; MAP_W * map_h];
    let mut mask_map = vec![0.0_f32; MAP_W * map_h];
    let mut total_weight = 0.0_f32;
    for my in 0..map_h {
        for mx in 0..MAP_W {
            let nx = (mx as f32 / (MAP_W - 1) as f32) * x_range + x_lo;
            let ny = (my as f32 / (map_h - 1) as f32) * y_range + y_lo;
            let d = depth_at(nx, ny, geo);
            let m = mask_at(nx, ny, geo);
            let i = my * MAP_W + mx;
            depth_map[i] = d;
            mask_map[i] = m;
            total_weight += m;
        }
    }

    // Inverse-CDF table over the mask grid so we can sample in O(log N).
    let mut cdf = Vec::with_capacity(MAP_W * map_h);
    let mut running = 0.0_f32;
    for &m in &mask_map {
        running += m;
        cdf.push(running);
    }

    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // Avatar uses ~92% face / 8% ambient.
        if rng.f32() < 0.92 {
            let target_weight = rng.f32() * total_weight;
            // Binary search the CDF.
            let mut lo = 0usize;
            let mut hi = cdf.len();
            while lo < hi {
                let mid = (lo + hi) / 2;
                if cdf[mid] < target_weight {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let idx = lo.min(cdf.len() - 1);
            let mx = idx % MAP_W;
            let my = idx / MAP_W;

            // Sub-pixel jitter so particles don't grid-align.
            let jx = (rng.f32() - 0.5) / MAP_W as f32;
            let jy = (rng.f32() - 0.5) / map_h as f32;
            let nx = (mx as f32 / (MAP_W - 1) as f32) * x_range + x_lo + jx;
            let ny = (my as f32 / (map_h - 1) as f32) * y_range + y_lo + jy;
            let depth = depth_at(nx, ny, geo);

            // Z = depth + thin shell noise.
            let z = depth + (rng.f32() - 0.5) * 0.03;

            let color = color_point(nx, ny, depth, pal, geo, &mut rng);

            // Mark eye-region particles so the renderer can give them
            // a per-frame flicker. The threshold matches the gaussian
            // sigmas in colorPoint: anything with strong eye-blob
            // density gets the fire flicker.
            let eye_sig = 0.045 * geo.eye_size;
            let eye_sig_y = 0.028 * geo.eye_size;
            let dx_l = nx - (-geo.eye_spread);
            let dx_r = nx - geo.eye_spread;
            let dy = ny - 0.22;
            let eye_l_g = (-(dx_l * dx_l) / (2.0 * eye_sig * eye_sig)
                - (dy * dy) / (2.0 * eye_sig_y * eye_sig_y))
                .exp();
            let eye_r_g = (-(dx_r * dx_r) / (2.0 * eye_sig * eye_sig)
                - (dy * dy) / (2.0 * eye_sig_y * eye_sig_y))
                .exp();
            let is_eye = eye_l_g.max(eye_r_g) > 0.4;

            out.push(Particle {
                target: [
                    nx * scale * geo.aspect_x,
                    ny * scale * geo.aspect_y,
                    z * scale,
                ],
                color,
                seed: rng.f32(),
                is_eye,
            });
        } else {
            // Ambient — a star around the face.
            let ang = rng.f32() * std::f32::consts::TAU;
            let dist = 0.6 + rng.f32() * 1.5;
            out.push(Particle {
                target: [
                    ang.cos() * dist * scale * 0.5,
                    (rng.f32() - 0.3) * scale,
                    (rng.f32() - 0.5) * scale * 0.3,
                ],
                color: [
                    pal.deep[0] * 0.3,
                    pal.deep[1] * 0.3,
                    pal.deep[2] * 0.3,
                    0.02 + rng.f32() * 0.05,
                ],
                seed: rng.f32(),
                is_eye: false,
            });
        }
    }

    out
}
