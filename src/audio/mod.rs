//! Voice-effects DSP chain — a Rust port of the per-character voice
//! processing in `~/src/avatar` (static/chronoflow/chronoflow-engine.js
//! and static/worklets/*.js).
//!
//! ## Why this exists
//!
//! Each character in the avatar app gets a distinct voice colour
//! applied to their ElevenLabs TTS output: Narrator clean, Utopia
//! bright/lifted (a fae voice), Oblivion dark/lowered/distorted (a
//! monolith voice). The colouring is purely DSP — same TTS source,
//! different effects chain.
//!
//! On the ghostly side the effects chain has to run server-side so it
//! can colour the voice BEFORE it gets mixed into the agent's AV
//! stream broadcast over MoQ. This module is the equivalent of the
//! avatar's WebAudio graph, ported to Rust + real-time-safe so it can
//! sit in the live audio path.
//!
//! ## Chain order
//!
//! Matches the JS: `input → EQ → compressor-in → formant → pitch →
//! [parallel: glitch, freqshift, comb, bitcrush] → mix → delay →
//! shimmer reverb → chorus → compressor-out → output`.
//!
//! ## Public API
//!
//! ```no_run
//! use ghostly::audio::{VoiceChain, profile};
//!
//! // Build a chain for Oblivion at 44.1 kHz mono.
//! let mut chain = VoiceChain::new(profile::for_character("oblivion"), 44_100.0);
//!
//! // Process a buffer of incoming TTS audio (typical chunk: 1024-4096 samples).
//! let mut buf = vec![0.0_f32; 1024];
//! chain.process(&mut buf);
//! ```
//!
//! The chain is stateful — keep one instance per voice stream. Call
//! [`VoiceChain::set_profile`] to switch character/emotion (the JS does
//! this on emotion transitions; the values snap, no crossfade yet).

pub mod nodes;
pub mod profile;

use nodes::{
    Bitcrusher, Chorus, Comb, Compressor, Delay, Eq3, FormantShifter, FreqShifter, Glitch,
    PitchShifter, Shimmer,
};
use profile::VoiceProfile;

/// A complete per-stream voice effects chain. All state is owned by
/// this struct so a renderer can hold one per active voice and call
/// [`Self::process`] from a real-time thread without allocations.
///
/// Internally lays out one instance of each effect node in the JS
/// chain order. Parameters are applied via [`Self::set_profile`] which
/// snaps every node's parameter set to the new profile's values.
pub struct VoiceChain {
    sample_rate: f32,
    eq: Eq3,
    comp_in: Compressor,
    formant: FormantShifter,
    pitch: PitchShifter,
    glitch: Glitch,
    freqshift: FreqShifter,
    comb: Comb,
    crush: Bitcrusher,
    delay: Delay,
    shimmer: Shimmer,
    chorus: Chorus,
    comp_out: Compressor,
}

impl VoiceChain {
    /// Build a fresh chain pre-loaded with `profile` at the given
    /// sample rate. Internal buffers (delay lines, granular pitch
    /// buffer, glitch capture buffer, shimmer reverb tail) size off
    /// the sample rate, so the chain is locked to it from this point.
    pub fn new(profile: VoiceProfile, sample_rate: f32) -> Self {
        let mut chain = Self {
            sample_rate,
            eq: Eq3::new(sample_rate),
            comp_in: Compressor::new(sample_rate, -18.0, 4.0, 0.003, 0.15, 10.0, 1.5),
            formant: FormantShifter::new(sample_rate),
            pitch: PitchShifter::new(sample_rate),
            glitch: Glitch::new(sample_rate),
            freqshift: FreqShifter::new(sample_rate),
            comb: Comb::new(sample_rate),
            crush: Bitcrusher::new(),
            delay: Delay::new(sample_rate, 0.3),
            shimmer: Shimmer::new(sample_rate),
            chorus: Chorus::new(sample_rate),
            comp_out: Compressor::new(sample_rate, -24.0, 8.0, 0.003, 0.10, 10.0, 4.0),
        };
        chain.set_profile(profile);
        chain
    }

    /// Snap every node's parameters to `profile`. Safe to call from
    /// the audio thread — no allocations. The JS version smoothly
    /// crossfades between profiles over ~600 ms on emotion change; we
    /// snap for now and can layer interpolation on top later if the
    /// transitions feel abrupt.
    pub fn set_profile(&mut self, p: VoiceProfile) {
        // EQ — 3-band shelving/peaking matching the JS defaults.
        self.eq.set(p.eq_low, p.eq_mid, p.eq_high);
        // Formant + pitch share the JS "voice spectral shaping" stage.
        self.formant.set(p.formant_shift, p.formant_mix);
        self.pitch.set(p.pitch_semitones, p.pitch_cents, p.pitch_mix);
        // Parallel destruction nodes — mixed back in via per-node mix.
        self.glitch
            .set(p.glitch_active, p.glitch_rate, p.glitch_size, p.glitch_prob, p.glitch_mix);
        self.freqshift.set(p.freq_shift_hz, p.freq_shift_mix);
        self.comb.set(p.comb_freq, p.comb_feedback, p.comb_mix);
        self.crush.set(p.crush_bits, p.crush_mix);
        // Time-based — sweetening at the end of the chain.
        self.delay.set(0.3, p.delay_feedback, p.delay_mix);
        self.shimmer
            .set(p.shimmer_decay, p.shimmer_amount, p.shimmer_mix);
        self.chorus.set(p.chorus_rate, 0.5, 3, p.chorus_mix);
    }

    /// Sample rate the chain was constructed with.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Process a mono f32 buffer in place. Real-time-safe — no
    /// allocations, no thread sync, all node state lives on the chain.
    pub fn process(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.step(*s);
        }
    }

    /// Process a single sample through the full chain. Exposed for
    /// callers that already have a sample loop (e.g. an encoder).
    #[inline]
    pub fn step(&mut self, input: f32) -> f32 {
        let mut s = input;
        s = self.eq.step(s);
        s = self.comp_in.step(s);
        s = self.formant.step(s);
        s = self.pitch.step(s);
        // Parallel destruction stack — each node mixes its own wet
        // signal in via internal `mix`. Series wiring is what the JS
        // graph does at this stage too (the "parallel" label in the
        // engine refers to the fact that each node has its own dry
        // bypass, not to parallel summing).
        s = self.glitch.step(s);
        s = self.freqshift.step(s);
        s = self.comb.step(s);
        s = self.crush.step(s);
        s = self.delay.step(s);
        s = self.shimmer.step(s);
        s = self.chorus.step(s);
        s = self.comp_out.step(s);
        s
    }
}
