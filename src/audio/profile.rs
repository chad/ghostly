//! Voice-effect profiles — originally ported verbatim from
//! `~/src/avatar/static/presenter.html` (VOICE_PROFILES, lines 466-555)
//! and the character→emotion mapping (line 967):
//!
//!   `narrator → calm, utopia → joy, oblivion → passion`
//!
//! DIVERGED from the avatar (2026-06-10): the wet mixes were tuned for
//! headphone theater; in live freeq calls they buried the words. After
//! repeated "I can't understand her" reports the time-based wets
//! (delay/shimmer/chorus), freq-shift beating, and pitch extremes were
//! cut roughly in half across every profile — A/B'd live on olive
//! (CURIOSITY) first, then the same ratios applied to the rest. The
//! character color is still there; the consonants survive now. If you
//! re-port from the avatar, re-apply this haircut.

use serde::{Deserialize, Serialize};

/// One frozen state of the voice chain. Mirrors the JS profile object
/// 1:1; field order matches the `applyVoiceProfile()` calls in
/// presenter.html so the port is auditable side-by-side.
///
/// Derives `Serialize`/`Deserialize` so the *entire* voice identity can
/// be carried in a [`crate::pack::CharacterPack`] and forked as data.
/// The `label` is a `&'static str` (can't be deserialized into), so it's
/// skipped on the wire and defaults to `"custom"` when loaded from a pack.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VoiceProfile {
    /// Human-friendly tag — shown in the avatar's emotion overlay.
    #[serde(skip, default = "default_voice_label")]
    pub label: &'static str,
    // ── Spectral shaping ─────────────────────────────────────────
    pub formant_shift: f32,
    pub formant_mix: f32,
    pub pitch_semitones: f32,
    pub pitch_cents: f32,
    pub pitch_mix: f32,
    // ── EQ (lowshelf @ 100 Hz / peak @ 1 kHz / highshelf @ 8 kHz) ─
    pub eq_low: f32,
    pub eq_mid: f32,
    pub eq_high: f32,
    // ── Parallel destruction ─────────────────────────────────────
    pub glitch_active: bool,
    pub glitch_rate: f32,
    pub glitch_size: f32,
    pub glitch_prob: f32,
    pub glitch_mix: f32,
    pub freq_shift_hz: f32,
    pub freq_shift_mix: f32,
    pub comb_freq: f32,
    pub comb_feedback: f32,
    pub comb_mix: f32,
    pub crush_bits: f32,
    pub crush_mix: f32,
    // ── Time/space sweetening ────────────────────────────────────
    pub delay_mix: f32,
    pub delay_feedback: f32,
    pub shimmer_mix: f32,
    pub shimmer_decay: f32,
    pub shimmer_amount: f32,
    pub chorus_mix: f32,
    pub chorus_rate: f32,
    /// Per-profile final gain applied AFTER the output compressor.
    /// Linear (1.0 = pass-through). Compensates for the level loss a
    /// heavy wet chain (PASSION: full formant + full pitch + delay +
    /// shimmer + chorus) suffers compared to a near-dry chain (CALM,
    /// where most node mixes are 0). Tune by ear against the dry
    /// reference voice — the goal is that switching characters does
    /// not require touching the listener's volume.
    pub output_gain: f32,
}

impl VoiceProfile {
    /// Neutral fallback — silent on every effect. Useful as a
    /// safety default and as the "off" target the JS code uses when
    /// it transitions away from an emotion.
    pub const NEUTRAL: VoiceProfile = CALM;
}

/// Label restored for voice profiles deserialized from a pack (the
/// real `&'static str` label isn't carried on the wire).
fn default_voice_label() -> &'static str {
    "custom"
}

/// ✧ Luminous / Fae — UTOPIA. Lifted formants, slight detune up,
/// crystalline shimmer.
///
/// Dialled back from the avatar's `+4 / +2` to `+1.5 / +1` after the
/// live A/B — the original lift pushed Utopia's natural register too
/// high (chipmunk territory). The lighter touch keeps the sense of
/// warmth and brightness without taking the speaker's own voice out
/// from under it. Shimmer + chorus carry the rest of the sparkle.
pub const JOY: VoiceProfile = VoiceProfile {
    label: "joy",
    formant_shift: 1.5,
    formant_mix: 0.8,
    pitch_semitones: 1.0,
    pitch_cents: 8.0,
    pitch_mix: 1.0,
    eq_low: -1.0,
    eq_mid: 1.0,
    eq_high: 2.0,
    glitch_active: false,
    glitch_rate: 6.0,
    glitch_size: 0.08,
    glitch_prob: 0.3,
    glitch_mix: 0.0,
    freq_shift_hz: 4.0,
    freq_shift_mix: 0.03,
    comb_freq: 300.0,
    comb_feedback: 0.3,
    comb_mix: 0.0,
    crush_bits: 16.0,
    crush_mix: 0.0,
    delay_mix: 0.04,
    delay_feedback: 0.12,
    shimmer_mix: 0.08,
    shimmer_decay: 1.5,
    shimmer_amount: 0.2,
    chorus_mix: 0.1,
    chorus_rate: 1.8,
    // Drier sparkle loses less level than the original full wash.
    output_gain: 1.2,
};

/// ◌ Spectral / Diffuse — distant, dreamy. These are olive's live-tuned
/// numbers (the "ghostly-olive.json" A/B on 2026-06-09): barely-lowered
/// pitch, a whisper of shimmer/chorus, no comb. Reads as the same
/// spectral character but every word lands.
pub const CURIOSITY: VoiceProfile = VoiceProfile {
    label: "curiosity",
    formant_shift: 0.0,
    formant_mix: 1.0,
    pitch_semitones: -0.5,
    pitch_cents: -8.0,
    pitch_mix: 1.0,
    eq_low: 1.0,
    eq_mid: 1.0,
    eq_high: 1.0,
    glitch_active: false,
    glitch_rate: 4.0,
    glitch_size: 0.1,
    glitch_prob: 0.2,
    glitch_mix: 0.0,
    freq_shift_hz: 6.0,
    freq_shift_mix: 0.03,
    comb_freq: 250.0,
    comb_feedback: 0.35,
    comb_mix: 0.0,
    crush_bits: 16.0,
    crush_mix: 0.0,
    delay_mix: 0.05,
    delay_feedback: 0.15,
    shimmer_mix: 0.08,
    shimmer_decay: 1.5,
    shimmer_amount: 0.3,
    chorus_mix: 0.08,
    chorus_rate: 0.7,
    // Much drier chain loses less level — modest makeup only.
    output_gain: 1.15,
};

/// ▲ Monolith / Force — OBLIVION. Dark, lowered, mechanical edge.
/// Dialled down from the avatar's `-4 / -3` to `-2.5 / -1` after the
/// JS values were over-deepened in the live freeq broadcast — too
/// far into cartoon-demon territory. Still reads as Oblivion (darker
/// than baseline, faint mechanical glaze from the comb + bitcrush)
/// without crossing into novelty.
pub const PASSION: VoiceProfile = VoiceProfile {
    label: "passion",
    formant_shift: -1.8,
    formant_mix: 0.8,
    pitch_semitones: -1.0,
    pitch_cents: 0.0,
    pitch_mix: 1.0,
    eq_low: 3.0,
    eq_mid: 1.0,
    eq_high: -1.0,
    glitch_active: false,
    glitch_rate: 12.0,
    glitch_size: 0.04,
    glitch_prob: 0.2,
    glitch_mix: 0.0,
    freq_shift_hz: 8.0,
    freq_shift_mix: 0.02,
    comb_freq: 150.0,
    comb_feedback: 0.25,
    comb_mix: 0.02,
    crush_bits: 14.0,
    crush_mix: 0.01,
    delay_mix: 0.05,
    delay_feedback: 0.2,
    shimmer_mix: 0.05,
    shimmer_decay: 2.0,
    shimmer_amount: 0.15,
    chorus_mix: 0.05,
    chorus_rate: 0.4,
    // Still the wettest chain, but at half the old mixes it loses far
    // less level than the original (which needed +7 dB makeup).
    output_gain: 1.6,
};

/// ○ Liminal / Still — NARRATOR. Near-identity, gentle sweetening.
pub const CALM: VoiceProfile = VoiceProfile {
    label: "calm",
    formant_shift: 0.0,
    formant_mix: 0.0,
    pitch_semitones: 0.0,
    pitch_cents: 0.0,
    pitch_mix: 0.0,
    eq_low: 0.0,
    eq_mid: 0.0,
    eq_high: 0.0,
    glitch_active: false,
    glitch_rate: 4.0,
    glitch_size: 0.1,
    glitch_prob: 0.2,
    glitch_mix: 0.0,
    freq_shift_hz: 0.0,
    freq_shift_mix: 0.0,
    comb_freq: 200.0,
    comb_feedback: 0.0,
    comb_mix: 0.0,
    crush_bits: 16.0,
    crush_mix: 0.0,
    delay_mix: 0.03,
    delay_feedback: 0.1,
    shimmer_mix: 0.05,
    shimmer_decay: 1.2,
    shimmer_amount: 0.15,
    chorus_mix: 0.06,
    chorus_rate: 0.4,
    // Baseline — CALM bypasses formant + pitch + most destruction
    // nodes (mix=0), so the chain is near-identity and needs no
    // additional makeup.
    output_gain: 1.0,
};

/// ◆ Abyssal / Deity.
pub const AWE: VoiceProfile = VoiceProfile {
    label: "awe",
    formant_shift: -2.0,
    formant_mix: 0.5,
    pitch_semitones: -1.0,
    pitch_cents: -5.0,
    pitch_mix: 1.0,
    eq_low: 3.0,
    eq_mid: 2.0,
    eq_high: 1.0,
    glitch_active: false,
    glitch_rate: 3.0,
    glitch_size: 0.12,
    glitch_prob: 0.25,
    glitch_mix: 0.0,
    freq_shift_hz: 5.0,
    freq_shift_mix: 0.02,
    comb_freq: 160.0,
    comb_feedback: 0.15,
    comb_mix: 0.02,
    crush_bits: 16.0,
    crush_mix: 0.0,
    delay_mix: 0.1,
    delay_feedback: 0.2,
    shimmer_mix: 0.12,
    shimmer_decay: 2.5,
    shimmer_amount: 0.3,
    chorus_mix: 0.1,
    chorus_rate: 0.3,
    output_gain: 1.3,
};

/// ♡ Hearthside / Close.
pub const WARMTH: VoiceProfile = VoiceProfile {
    label: "warmth",
    formant_shift: -1.5,
    formant_mix: 1.0,
    pitch_semitones: -1.0,
    pitch_cents: 0.0,
    pitch_mix: 1.0,
    eq_low: 2.0,
    eq_mid: 1.0,
    eq_high: -1.0,
    glitch_active: false,
    glitch_rate: 4.0,
    glitch_size: 0.1,
    glitch_prob: 0.2,
    glitch_mix: 0.0,
    freq_shift_hz: 0.0,
    freq_shift_mix: 0.0,
    comb_freq: 250.0,
    comb_feedback: 0.3,
    comb_mix: 0.0,
    crush_bits: 16.0,
    crush_mix: 0.0,
    delay_mix: 0.06,
    delay_feedback: 0.2,
    shimmer_mix: 0.06,
    shimmer_decay: 2.5,
    shimmer_amount: 0.2,
    chorus_mix: 0.08,
    chorus_rate: 0.5,
    output_gain: 1.25,
};

/// ★ Stadium / Presence.
pub const TRIUMPH: VoiceProfile = VoiceProfile {
    label: "triumph",
    formant_shift: -1.5,
    formant_mix: 1.0,
    pitch_semitones: 0.0,
    pitch_cents: 0.0,
    pitch_mix: 1.0,
    eq_low: 2.0,
    eq_mid: 4.0,
    eq_high: 3.0,
    glitch_active: false,
    glitch_rate: 3.0,
    glitch_size: 0.1,
    glitch_prob: 0.2,
    glitch_mix: 0.0,
    freq_shift_hz: 0.0,
    freq_shift_mix: 0.0,
    comb_freq: 200.0,
    comb_feedback: 0.0,
    comb_mix: 0.0,
    crush_bits: 16.0,
    crush_mix: 0.0,
    delay_mix: 0.04,
    delay_feedback: 0.1,
    shimmer_mix: 0.05,
    shimmer_decay: 1.5,
    shimmer_amount: 0.15,
    chorus_mix: 0.05,
    chorus_rate: 0.4,
    output_gain: 1.2,
};

/// ▽ Void / Machine.
pub const CONCERN: VoiceProfile = VoiceProfile {
    label: "concern",
    formant_shift: -2.0,
    formant_mix: 1.0,
    pitch_semitones: -1.0,
    pitch_cents: -5.0,
    pitch_mix: 1.0,
    eq_low: 2.0,
    eq_mid: 0.0,
    eq_high: -1.0,
    glitch_active: false,
    glitch_rate: 8.0,
    glitch_size: 0.03,
    glitch_prob: 0.2,
    glitch_mix: 0.0,
    freq_shift_hz: 5.0,
    freq_shift_mix: 0.02,
    comb_freq: 180.0,
    comb_feedback: 0.2,
    comb_mix: 0.02,
    crush_bits: 14.0,
    crush_mix: 0.01,
    delay_mix: 0.06,
    delay_feedback: 0.2,
    shimmer_mix: 0.05,
    shimmer_decay: 2.0,
    shimmer_amount: 0.15,
    chorus_mix: 0.06,
    chorus_rate: 0.3,
    output_gain: 1.5,
};

/// Look up an emotion profile by JS name. Returns [`CALM`] if the
/// name doesn't match — mirrors the JS `VOICE_PROFILES[emotion] ||
/// VOICE_PROFILES.calm` fallback.
pub fn for_emotion(name: &str) -> VoiceProfile {
    match name {
        "joy" => JOY,
        "curiosity" => CURIOSITY,
        "passion" => PASSION,
        "calm" => CALM,
        "awe" => AWE,
        "warmth" => WARMTH,
        "triumph" => TRIUMPH,
        "concern" => CONCERN,
        _ => CALM,
    }
}

/// Look up the default voice profile for a character name. Mirrors
/// the avatar's character→emotion map in presenter.html line 967.
/// Unknown characters fall through to [`CALM`].
pub fn for_character(name: &str) -> VoiceProfile {
    match name {
        "narrator" => CALM,
        "utopia" => JOY,
        "oblivion" => PASSION,
        // Eliza isn't defined in the avatar yet — pick something
        // close to her teal placeholder palette feel.
        "eliza" => CURIOSITY,
        _ => CALM,
    }
}
