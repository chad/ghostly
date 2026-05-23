//! Sentiment-driven palette morphing.
//!
//! Ported from the JS `setEmotion` / `EMOTION_STYLE_MAP` /
//! `EMOTION_STYLE_OVERRIDES` in `face-of-god.js` + `engine.js`. The
//! avatar reference reacts to a stream of `Emotion` events over a
//! WebSocket: each one nudges the rendering toward the matching
//! "style" (a named palette + intensity bundle), interpolated over a
//! few hundred ms so transitions feel cinematic rather than snappy.
//!
//! In ghostly the analogue is [`apply_emotion`] — given a fresh
//! [`Character`] and an `Emotion + intensity`, it returns a new
//! Character whose [`Palette`] + [`RenderConfig`] have been blended
//! toward the emotion's mood. Renderers call this each time the
//! emotion changes (driven by the host application — e.g.
//! freeq-eliza's mood detection, or the avatar's keyword sentiment
//! classifier).
//!
//! Eight emotions matched to mood "styles":
//!
//! | Emotion | Mood style | Colour family |
//! | --- | --- | --- |
//! | Joy | solar | warm yellow-gold |
//! | Triumph | solar++ | bright golden white |
//! | Curiosity | nebula | electric blue-violet |
//! | Passion | inferno | hot orange-red |
//! | Calm | ghost | strong blue |
//! | Awe | abyss | deep ocean blue |
//! | Warmth | solar (mild) | warm amber |
//! | Concern | rage (mild) | dark crimson |
//!
//! These styles match the avatar JS verbatim — porting them lets a
//! sentiment classifier built for the avatar drive ghostly without
//! re-tuning.

use crate::character::{Character, Palette, RenderConfig};

/// The eight emotions ghostly recognises — taken from the avatar
/// sentiment classifier output set in `face-of-god.js`. The host app
/// classifies speech / chat and forwards the resulting `Emotion`
/// here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emotion {
    Joy,
    Triumph,
    Curiosity,
    Passion,
    Calm,
    Awe,
    Warmth,
    Concern,
}

impl Emotion {
    /// Parse a lowercase emotion name (e.g. `"joy"`). `None` for any
    /// unknown string — host apps should treat that as Calm.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "joy" => Emotion::Joy,
            "triumph" => Emotion::Triumph,
            "curiosity" => Emotion::Curiosity,
            "passion" => Emotion::Passion,
            "calm" => Emotion::Calm,
            "awe" => Emotion::Awe,
            "warmth" => Emotion::Warmth,
            "concern" => Emotion::Concern,
            _ => return None,
        })
    }

    /// Per-emotion palette + render-tweak bundle. Pure data, easy to
    /// tune.
    fn target(self) -> EmotionTarget {
        match self {
            Emotion::Joy => EmotionTarget {
                palette: solar_palette(),
                fresnel_intensity: 0.55,
                glow_boost: 1.1,
            },
            Emotion::Triumph => EmotionTarget {
                // Pushed further than Joy — brighter highlights, more
                // saturation, more rim glow.
                palette: Palette {
                    base: [1.0, 0.92, 0.55],
                    deep: [0.6, 0.45, 0.15],
                    highlight: [1.0, 1.0, 0.85],
                    eye: [1.0, 0.95, 0.4],
                    eye_rim: None,
                    glow: [1.0, 0.95, 0.55],
                },
                fresnel_intensity: 0.75,
                glow_boost: 1.5,
            },
            Emotion::Curiosity => EmotionTarget {
                // Electric blue-violet — alert + slightly cool.
                palette: Palette {
                    base: [0.4, 0.55, 1.0],
                    deep: [0.05, 0.10, 0.35],
                    highlight: [0.75, 0.85, 1.0],
                    eye: [0.6, 0.85, 1.0],
                    eye_rim: None,
                    glow: [0.4, 0.6, 1.0],
                },
                fresnel_intensity: 0.55,
                glow_boost: 1.05,
            },
            Emotion::Passion => EmotionTarget {
                palette: inferno_palette(),
                fresnel_intensity: 0.65,
                glow_boost: 1.3,
            },
            Emotion::Calm => EmotionTarget {
                palette: Palette::GHOST,
                fresnel_intensity: 0.5,
                glow_boost: 1.0,
            },
            Emotion::Awe => EmotionTarget {
                // Deep abyssal blue — silent, immense.
                palette: Palette {
                    base: [0.25, 0.40, 0.95],
                    deep: [0.02, 0.05, 0.20],
                    highlight: [0.55, 0.70, 1.0],
                    eye: [0.4, 0.6, 1.0],
                    eye_rim: None,
                    glow: [0.20, 0.40, 1.0],
                },
                fresnel_intensity: 0.45,
                glow_boost: 0.95,
            },
            Emotion::Warmth => EmotionTarget {
                // Solar but mild — a smaller intensity scale than Joy.
                palette: Palette {
                    base: [0.85, 0.55, 0.30],
                    deep: [0.35, 0.20, 0.08],
                    highlight: [1.0, 0.80, 0.50],
                    eye: [1.0, 0.7, 0.3],
                    eye_rim: None,
                    glow: [1.0, 0.6, 0.25],
                },
                fresnel_intensity: 0.50,
                glow_boost: 1.05,
            },
            Emotion::Concern => EmotionTarget {
                // Mild Rage — darker, more saturated red, less glow.
                palette: Palette {
                    base: [0.55, 0.10, 0.10],
                    deep: [0.15, 0.02, 0.02],
                    highlight: [0.85, 0.20, 0.15],
                    eye: [0.95, 0.30, 0.10],
                    eye_rim: Some([1.0, 0.10, 0.0]),
                    glow: [0.85, 0.10, 0.05],
                },
                fresnel_intensity: 0.55,
                glow_boost: 1.10,
            },
        }
    }
}

struct EmotionTarget {
    palette: Palette,
    fresnel_intensity: f32,
    glow_boost: f32,
}

fn solar_palette() -> Palette {
    Palette {
        base: [1.0, 0.78, 0.30],
        deep: [0.50, 0.32, 0.10],
        highlight: [1.0, 0.95, 0.60],
        eye: [1.0, 0.85, 0.30],
        eye_rim: None,
        glow: [1.0, 0.75, 0.25],
    }
}

fn inferno_palette() -> Palette {
    Palette {
        base: [1.0, 0.40, 0.15],
        deep: [0.45, 0.10, 0.03],
        highlight: [1.0, 0.65, 0.30],
        eye: [1.0, 0.5, 0.1],
        eye_rim: Some([1.0, 0.20, 0.0]),
        glow: [1.0, 0.30, 0.10],
    }
}

/// Blend `character` toward `emotion` by `intensity` (`0..=1`). `0`
/// returns the character unchanged; `1` is the full emotion target.
/// The host calls this each time the detected emotion changes — the
/// returned `Character` replaces the live one (or, for true crossfade,
/// the host can render two and alpha-blend the pixmaps).
///
/// Note: the [`crate::character::Transition`] closure isn't blended —
/// it's a behaviour rather than a colour, and crossfading two
/// behaviours produces uncanny motion. The transition fires on
/// scatter, and a scatter event is a natural moment to swap the
/// character outright.
pub fn apply_emotion(character: &Character, emotion: Emotion, intensity: f32) -> Character {
    let t = intensity.clamp(0.0, 1.0);
    let tgt = emotion.target();
    let palette = lerp_palette(&character.palette, &tgt.palette, t);

    // Blend select RenderConfig fields. Most config is character
    // identity (geometry-bound), but rim glow + a soft contour-alpha
    // boost feel emotional.
    let mut render_config = character.render_config.clone();
    render_config.fresnel_intensity = lerp(
        character.render_config.fresnel_intensity,
        tgt.fresnel_intensity,
        t,
    );

    // Boost contour alpha + breath for high-emotion states. We hold
    // the original baseline shape (line widths) and only nudge the
    // perceived "brightness".
    let mut contour_baseline = character.contour_baseline;
    contour_baseline.outer_alpha = (contour_baseline.outer_alpha * (1.0 + (tgt.glow_boost - 1.0) * t)).min(1.0);
    contour_baseline.inner_alpha = (contour_baseline.inner_alpha * (1.0 + (tgt.glow_boost - 1.0) * t)).min(1.0);

    Character {
        name: character.name,
        geometry: character.geometry,
        palette,
        contour_color: character.contour_color,
        contour_paths: character.contour_paths.clone(),
        transition: rebuild_transition(&character.transition),
        contour_baseline,
        render_config,
    }
}

/// Cheap palette interpolation. Each channel of each role lerps
/// independently. `eye_rim` stays whichever side has it (`Some`
/// wins) — it's the fire-eye flag, not a colour to blend.
fn lerp_palette(a: &Palette, b: &Palette, t: f32) -> Palette {
    Palette {
        base: lerp_rgb(a.base, b.base, t),
        deep: lerp_rgb(a.deep, b.deep, t),
        highlight: lerp_rgb(a.highlight, b.highlight, t),
        eye: lerp_rgb(a.eye, b.eye, t),
        eye_rim: match (a.eye_rim, b.eye_rim) {
            (Some(ra), Some(rb)) => Some(lerp_rgb(ra, rb, t)),
            (Some(r), None) | (None, Some(r)) if t < 0.5 => {
                // Hold the original side until past the midpoint.
                if a.eye_rim.is_some() && t < 0.5 {
                    Some(r)
                } else {
                    None
                }
            }
            _ => a.eye_rim,
        },
        glow: lerp_rgb(a.glow, b.glow, t),
    }
}

#[inline]
fn lerp_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// `Transition` carries a boxed closure that doesn't implement
/// `Clone`, so `apply_emotion` rebuilds a transition with the same
/// enter/exit speeds and a *default* displace closure. In practice the
/// host calls `apply_emotion` and then either keeps the existing
/// `FaceState` (which already has scatter destinations from the
/// character's original displace) or scatters again at which point the
/// character is re-built from scratch anyway.
fn rebuild_transition(original: &crate::character::Transition) -> crate::character::Transition {
    crate::character::Transition {
        // Default displace as a sane fallback — see the note above.
        displace: Box::new(|fnx, fny, ease, _time, seed| {
            let ang = (fny - 0.1).atan2(fnx) + seed * 0.5;
            let dist = ease * (2.0 + seed * 2.0);
            [
                ang.cos() * dist,
                ang.sin() * dist,
                (seed - 0.5) * ease * 2.0,
            ]
        }),
        enter_speed: original.enter_speed,
        exit_speed: original.exit_speed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::characters;

    #[test]
    fn parse_handles_known_emotions() {
        assert_eq!(Emotion::parse("joy"), Some(Emotion::Joy));
        assert_eq!(Emotion::parse("Triumph"), Some(Emotion::Triumph));
        assert_eq!(Emotion::parse("CONCERN"), Some(Emotion::Concern));
    }

    #[test]
    fn parse_rejects_unknown() {
        assert_eq!(Emotion::parse("euphoria"), None);
    }

    #[test]
    fn zero_intensity_is_identity() {
        // intensity=0 must leave the palette where it started — the
        // host might re-call apply_emotion every frame, so identity at
        // zero is load-bearing.
        let c = characters::by_name("narrator").unwrap();
        let blended = apply_emotion(&c, Emotion::Passion, 0.0);
        assert_eq!(blended.palette.base, c.palette.base);
        assert_eq!(blended.palette.deep, c.palette.deep);
    }

    #[test]
    fn full_intensity_lands_on_target() {
        let c = characters::by_name("narrator").unwrap();
        let blended = apply_emotion(&c, Emotion::Joy, 1.0);
        let expected = Emotion::Joy.target().palette.base;
        for (a, b) in blended.palette.base.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
