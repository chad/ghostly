//! Implicit face: a 2D depth map sampled in `[-0.5, 0.5] × [-0.65, 0.85]`.
//!
//! `depth_at` returns ~`0..0.55` (higher = more forward); `mask_at`
//! returns where particles should exist (`0..1`); `color_point` returns
//! the per-particle RGBA. All ported verbatim from `face-gen.js` so
//! every character renders identically to the avatar reference.

use crate::character::{Geometry, Palette};

#[inline]
fn smoothstep(lo: f32, hi: f32, t: f32) -> f32 {
    let x = ((t - lo) / (hi - lo)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[inline]
fn gauss(x: f32, center: f32, sigma: f32) -> f32 {
    let d = x - center;
    (-(d * d) / (2.0 * sigma * sigma)).exp()
}

#[inline]
fn gauss2d(x: f32, y: f32, cx: f32, cy: f32, sx: f32, sy: f32) -> f32 {
    gauss(x, cx, sx) * gauss(y, cy, sy)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Depth at a normalized point. `0` = nothing here; higher = more
/// forward. Adds contributions from brow, nose, eyes, cheekbones,
/// lips, chin, temples, forehead, plus optional devil horns.
pub fn depth_at(x: f32, y: f32, g: &Geometry) -> f32 {
    // Base ellipsoidal head shape.
    let head_x = x / g.head_width;
    let head_y = (y - 0.1) / g.head_height;
    let head_r = head_x * head_x + head_y * head_y;
    if head_r > 1.2 {
        if g.horns {
            return horn_depth_at(x, y);
        }
        return 0.0;
    }
    let head = (1.0 - head_r).max(0.0) * 0.42;
    let mut d = head;

    // Narrow jaw — only matters below brow line.
    if y < -0.1 {
        let jaw_amount = smoothstep(-0.1, -0.55, y);
        let jaw_width = g.head_width * (1.0 - jaw_amount * g.jaw_narrow);
        if x.abs() > jaw_width {
            return 0.0;
        }
    }

    // Brow ridge.
    d += gauss2d(x, y, 0.0, 0.3, 0.28, 0.04) * g.brow_ridge;

    // Nose bridge + tip + nostrils.
    let nose_profile = smoothstep(-0.12, 0.05, y) * smoothstep(0.32, 0.15, y);
    d += gauss(x, 0.0, 0.04) * nose_profile * 0.16 * g.nose_scale;
    d += gauss2d(x, y, 0.0, -0.06, 0.055 * g.nose_scale, 0.04) * 0.1 * g.nose_scale;
    d += gauss2d(x, y, -0.06, -0.08, 0.03, 0.025) * 0.04 * g.nose_scale;
    d += gauss2d(x, y, 0.06, -0.08, 0.03, 0.025) * 0.04 * g.nose_scale;

    // Eye sockets recess.
    let eye_sig_x = 0.065 * g.eye_size;
    let eye_sig_y = 0.04 * g.eye_size;
    d -= gauss2d(x, y, -g.eye_spread, 0.22, eye_sig_x, eye_sig_y) * g.eye_socket_depth;
    d -= gauss2d(x, y, g.eye_spread, 0.22, eye_sig_x, eye_sig_y) * g.eye_socket_depth;

    // Eyeballs protrude inside the sockets.
    let eb_sig_x = 0.04 * g.eye_size;
    let eb_sig_y = 0.025 * g.eye_size;
    d += gauss2d(x, y, -g.eye_spread, 0.22, eb_sig_x, eb_sig_y) * g.eyeball_bump;
    d += gauss2d(x, y, g.eye_spread, 0.22, eb_sig_x, eb_sig_y) * g.eyeball_bump;

    // Cheekbones.
    d += gauss2d(x, y, -0.28, 0.1, 0.1, 0.08) * g.cheek_bone;
    d += gauss2d(x, y, 0.28, 0.1, 0.1, 0.08) * g.cheek_bone;

    // Mouth depression + lips.
    d -= gauss2d(x, y, 0.0, -0.2, 0.13, 0.02) * 0.06;
    d += gauss2d(x, y, 0.0, -0.16, 0.1, 0.012 * g.lip_fullness) * 0.03 * g.lip_fullness;
    d += gauss2d(x, y, 0.0, -0.24, 0.1, 0.012 * g.lip_fullness) * 0.025 * g.lip_fullness;

    // Chin.
    d += gauss2d(x, y, 0.0, -0.4, 0.08, 0.06) * g.chin_size;

    // Temples + forehead.
    d -= gauss2d(x, y, -0.35, 0.35, 0.07, 0.1) * g.temple_indent;
    d -= gauss2d(x, y, 0.35, 0.35, 0.07, 0.1) * g.temple_indent;
    d += gauss2d(x, y, 0.0, 0.5, 0.25, 0.15) * g.forehead_curve;

    if g.horns {
        d += horn_depth_at(x, y);
    }

    d.max(0.0)
}

/// Devil horns: curved ridges rising from each temple, narrowing to
/// the tip. `t` runs `0..1` along each horn.
pub fn horn_depth_at(x: f32, y: f32) -> f32 {
    let mut d = 0.0;
    for side in [-1.0_f32, 1.0_f32] {
        let bx = side * 0.32;
        let by = 0.45;
        let dy = y - by;
        const HORN_LEN: f32 = 0.45;
        let t = (dy / HORN_LEN).max(0.0);
        if t > 1.0 {
            continue;
        }
        // Horn arcs outward as it rises.
        let horn_cx = bx + side * t * t * 0.18;
        let dist = (x - horn_cx).abs();
        let radius = 0.05 * (1.0 - t * 0.7);
        if dist < radius {
            let profile = 1.0 - dist / radius;
            d += profile * profile * 0.25 * (1.0 - t * 0.5);
        }
    }
    d
}

/// Where particles SHOULD live. Boosts density around features so eyes
/// + nose + mouth + brow read clearly even at low particle counts.
pub fn mask_at(x: f32, y: f32, g: &Geometry) -> f32 {
    let hw = g.head_width + 0.02;
    let hh = g.head_height + 0.03;

    // Head ellipse.
    let head_x = x / hw;
    let head_y = (y - 0.1) / hh;
    let head_r = head_x * head_x + head_y * head_y;

    // Horn membership.
    let mut in_horn = false;
    if g.horns {
        for side in [-1.0_f32, 1.0_f32] {
            let bx = side * 0.32;
            let by = 0.45;
            let dy = y - by;
            let t = dy / 0.45;
            if (0.0..=1.0).contains(&t) {
                let horn_cx = bx + side * t * t * 0.18;
                let radius = 0.07 * (1.0 - t * 0.6);
                if (x - horn_cx).abs() < radius {
                    in_horn = true;
                }
            }
        }
    }

    if head_r > 1.0 && !in_horn {
        return 0.0;
    }

    // Narrow jaw mask cut-off.
    if y < -0.1 && !in_horn {
        let jaw_amount = smoothstep(-0.1, -0.55, y);
        let jaw_width = hw * (1.0 - jaw_amount * g.jaw_narrow);
        if x.abs() > jaw_width {
            return 0.0;
        }
    }

    let mut mask = if in_horn {
        0.6
    } else {
        smoothstep(1.0, 0.7, head_r) * 0.5
    };

    // Feature density boosts.
    let esx = 0.08 * g.eye_size;
    let esy = 0.05 * g.eye_size;
    mask += gauss2d(x, y, -g.eye_spread, 0.22, esx, esy) * 0.7;
    mask += gauss2d(x, y, g.eye_spread, 0.22, esx, esy) * 0.7;
    mask += gauss(x, 0.0, 0.05)
        * smoothstep(-0.15, 0.0, y)
        * smoothstep(0.35, 0.15, y)
        * 0.3;
    mask += gauss2d(x, y, 0.0, -0.2, 0.14, 0.03) * 0.5;
    mask += gauss2d(x, y, 0.0, 0.3, 0.25, 0.04) * 0.3;
    mask += gauss2d(x, y, 0.0, -0.4, 0.1, 0.06) * 0.25;

    mask.min(1.0)
}

/// Per-particle colour. Depth shading + eye glow + horn tint, with a
/// small per-particle noise so the face shimmers instead of looking
/// pixel-perfect.
pub fn color_point(
    nx: f32,
    ny: f32,
    depth: f32,
    pal: &Palette,
    g: &Geometry,
    rng: &mut fastrand::Rng,
) -> [f32; 4] {
    let hw = g.head_width + 0.02;
    let hh = g.head_height + 0.03;

    // Depth → base + deep + highlight blend.
    let dt = smoothstep(0.0, 0.4, depth);
    let mut r = lerp(pal.deep[0], pal.base[0], dt);
    let mut gv = lerp(pal.deep[1], pal.base[1], dt);
    let mut b = lerp(pal.deep[2], pal.base[2], dt);
    if depth > 0.3 {
        let ht = smoothstep(0.3, 0.5, depth) * 0.5;
        r = lerp(r, pal.highlight[0], ht);
        gv = lerp(gv, pal.highlight[1], ht);
        b = lerp(b, pal.highlight[2], ht);
    }

    // Eye glow.
    let eb_sig = 0.04 * g.eye_size;
    let eye_l = gauss2d(nx, ny, -g.eye_spread, 0.22, eb_sig, 0.025 * g.eye_size);
    let eye_r = gauss2d(nx, ny, g.eye_spread, 0.22, eb_sig, 0.025 * g.eye_size);
    let eyeness = eye_l.max(eye_r);
    if eyeness > 0.1 {
        let et = (eyeness * 0.8 * g.eye_glow).min(1.0);
        if let Some(rim) = pal.eye_rim {
            // Fire eyes — hot core (white) → eye color → rim.
            let core_t = (eyeness * 2.0).min(1.0);
            let (fire_r, fire_g, fire_b) = if core_t > 0.6 {
                let t = (core_t - 0.6) / 0.4;
                (
                    lerp(pal.eye[0], 1.0, t),
                    lerp(pal.eye[1], 0.9, t),
                    lerp(pal.eye[2], 0.5, t),
                )
            } else {
                let t = core_t / 0.6;
                (
                    lerp(rim[0], pal.eye[0], t),
                    lerp(rim[1], pal.eye[1], t),
                    lerp(rim[2], pal.eye[2], t),
                )
            };
            let flicker = 0.8 + rng.f32() * 0.4;
            r = lerp(r, (fire_r * flicker).min(1.0), et);
            gv = lerp(gv, (fire_g * flicker).min(1.0), et);
            b = lerp(b, (fire_b * flicker).min(1.0), et);
        } else {
            r = lerp(r, pal.eye[0], et);
            gv = lerp(gv, pal.eye[1], et);
            b = lerp(b, pal.eye[2], et);
        }
    }

    // Horn tint — darker, more saturated near the tips.
    if g.horns {
        let hd = horn_depth_at(nx, ny);
        if hd > 0.01 {
            let ht = (hd * 5.0).min(1.0);
            r = lerp(r, pal.highlight[0] * 0.6, ht);
            gv = lerp(gv, pal.highlight[1] * 0.3, ht);
            b = lerp(b, pal.highlight[2] * 0.2, ht);
        }
    }

    // Per-particle noise — keeps the field from looking computed.
    let noise = 0.88 + rng.f32() * 0.24;
    r *= noise;
    gv *= noise;
    b *= noise;

    // Alpha: edge fade + depth + eye/horn boost.
    let edge_x = nx / hw;
    let edge_y = (ny - 0.1) / hh;
    let edge_r = edge_x * edge_x + edge_y * edge_y;
    let edge_fade = smoothstep(1.0, 0.6, edge_r);
    let mut alpha = (0.15 + dt * 0.45) * edge_fade;
    if eyeness > 0.15 {
        alpha = alpha.max(0.4 + eyeness * 0.5 * g.eye_glow);
    }
    if g.horns {
        let hd = horn_depth_at(nx, ny);
        if hd > 0.01 {
            alpha = alpha.max(0.3 + hd * 2.0);
        }
    }

    [r.clamp(0.0, 1.0), gv.clamp(0.0, 1.0), b.clamp(0.0, 1.0), alpha.clamp(0.0, 1.0)]
}
