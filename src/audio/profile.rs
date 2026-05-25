//! Voice-effect profiles — ported verbatim from
//! `~/src/avatar/static/presenter.html` (VOICE_PROFILES, lines 466-555)
//! and the character→emotion mapping (line 967):
//!
//!   `narrator → calm, utopia → joy, oblivion → passion`
//!
//! Every numeric value here is the JS literal. Don't tune in this
//! file — the avatar is the source of truth; tune there, then port
//! the diff back.

/// One frozen state of the voice chain. Mirrors the JS profile object
/// 1:1; field order matches the `applyVoiceProfile()` calls in
/// presenter.html so the port is auditable side-by-side.
#[derive(Clone, Copy, Debug)]
pub struct VoiceProfile {
    /// Human-friendly tag — shown in the avatar's emotion overlay.
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

/// ✧ Luminous / Fae — UTOPIA. Lifted formants, slight detune up,
/// crystalline shimmer.
pub const JOY: VoiceProfile = VoiceProfile {
    label: "joy",
    formant_shift: 4.0,
    formant_mix: 1.0,
    pitch_semitones: 2.0,
    pitch_cents: 15.0,
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
    freq_shift_mix: 0.08,
    comb_freq: 300.0,
    comb_feedback: 0.3,
    comb_mix: 0.0,
    crush_bits: 16.0,
    crush_mix: 0.0,
    delay_mix: 0.06,
    delay_feedback: 0.12,
    shimmer_mix: 0.15,
    shimmer_decay: 1.5,
    shimmer_amount: 0.3,
    chorus_mix: 0.2,
    chorus_rate: 1.8,
    // ~+3 dB to match the dry reference — JOY has full formant + full
    // pitch + shimmer + chorus, all of which average the signal down.
    output_gain: 1.4,
};

/// ◌ Spectral / Diffuse — distant, dreamy.
pub const CURIOSITY: VoiceProfile = VoiceProfile {
    label: "curiosity",
    formant_shift: 0.0,
    formant_mix: 1.0,
    pitch_semitones: -2.0,
    pitch_cents: -20.0,
    pitch_mix: 1.0,
    eq_low: 1.0,
    eq_mid: 0.0,
    eq_high: 0.0,
    glitch_active: false,
    glitch_rate: 4.0,
    glitch_size: 0.1,
    glitch_prob: 0.2,
    glitch_mix: 0.0,
    freq_shift_hz: 6.0,
    freq_shift_mix: 0.1,
    comb_freq: 250.0,
    comb_feedback: 0.35,
    comb_mix: 0.05,
    crush_bits: 16.0,
    crush_mix: 0.0,
    delay_mix: 0.2,
    delay_feedback: 0.35,
    shimmer_mix: 0.3,
    shimmer_decay: 5.0,
    shimmer_amount: 0.45,
    chorus_mix: 0.25,
    chorus_rate: 0.7,
    // ~+3 dB to compensate full formant + full pitch + heavy shimmer.
    output_gain: 1.4,
};

/// ▲ Monolith / Force — OBLIVION. Dark, lowered, mechanical edge.
/// Dialled down from the avatar's `-4 / -3` to `-2.5 / -1` after the
/// JS values were over-deepened in the live freeq broadcast — too
/// far into cartoon-demon territory. Still reads as Oblivion (darker
/// than baseline, faint mechanical glaze from the comb + bitcrush)
/// without crossing into novelty.
pub const PASSION: VoiceProfile = VoiceProfile {
    label: "passion",
    formant_shift: -2.5,
    formant_mix: 1.0,
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
    freq_shift_mix: 0.06,
    comb_freq: 150.0,
    comb_feedback: 0.25,
    comb_mix: 0.04,
    crush_bits: 14.0,
    crush_mix: 0.02,
    delay_mix: 0.08,
    delay_feedback: 0.2,
    shimmer_mix: 0.1,
    shimmer_decay: 2.0,
    shimmer_amount: 0.15,
    chorus_mix: 0.1,
    chorus_rate: 0.4,
    // ~+7 dB. The heaviest wet chain — full formant shift, full pitch
    // shift, comb + bitcrush + freqshift active, delay + shimmer +
    // chorus. The user A/B-tested live and reported Oblivion was
    // noticeably quieter than Narrator's dry path; this brings him
    // back into the same loudness ballpark without driving comp_out
    // into clipping.
    output_gain: 2.2,
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
    delay_mix: 0.04,
    delay_feedback: 0.1,
    shimmer_mix: 0.08,
    shimmer_decay: 1.2,
    shimmer_amount: 0.15,
    chorus_mix: 0.1,
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
    freq_shift_mix: 0.03,
    comb_freq: 160.0,
    comb_feedback: 0.15,
    comb_mix: 0.02,
    crush_bits: 16.0,
    crush_mix: 0.0,
    delay_mix: 0.22,
    delay_feedback: 0.35,
    shimmer_mix: 0.35,
    shimmer_decay: 6.0,
    shimmer_amount: 0.55,
    chorus_mix: 0.25,
    chorus_rate: 0.3,
    output_gain: 1.6,
};

/// ♡ Hearthside / Close.
pub const WARMTH: VoiceProfile = VoiceProfile {
    label: "warmth",
    formant_shift: -2.0,
    formant_mix: 1.0,
    pitch_semitones: -2.0,
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
    delay_mix: 0.12,
    delay_feedback: 0.2,
    shimmer_mix: 0.12,
    shimmer_decay: 2.5,
    shimmer_amount: 0.2,
    chorus_mix: 0.15,
    chorus_rate: 0.5,
    output_gain: 1.5,
};

/// ★ Stadium / Presence.
pub const TRIUMPH: VoiceProfile = VoiceProfile {
    label: "triumph",
    formant_shift: -2.0,
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
    delay_mix: 0.06,
    delay_feedback: 0.1,
    shimmer_mix: 0.08,
    shimmer_decay: 1.5,
    shimmer_amount: 0.15,
    chorus_mix: 0.08,
    chorus_rate: 0.4,
    output_gain: 1.4,
};

/// ▽ Void / Machine.
pub const CONCERN: VoiceProfile = VoiceProfile {
    label: "concern",
    formant_shift: -3.0,
    formant_mix: 1.0,
    pitch_semitones: -2.0,
    pitch_cents: -10.0,
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
    freq_shift_mix: 0.04,
    comb_freq: 180.0,
    comb_feedback: 0.2,
    comb_mix: 0.03,
    crush_bits: 14.0,
    crush_mix: 0.02,
    delay_mix: 0.12,
    delay_feedback: 0.2,
    shimmer_mix: 0.1,
    shimmer_decay: 2.0,
    shimmer_amount: 0.15,
    chorus_mix: 0.12,
    chorus_rate: 0.3,
    output_gain: 2.0,
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
