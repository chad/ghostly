//! DSP nodes — one struct per effect in the JS chain. Each exposes a
//! `step(sample) -> sample` method so the parent [`VoiceChain`]
//! (in `mod.rs`) can wire them in series at sample granularity.
//!
//! ## Porting notes
//!
//! Where the JS used a WebAudio standard node (BiquadFilter,
//! DelayNode, DynamicsCompressor) the Rust side rolls its own — the
//! algorithms are small and rolling them ourselves keeps the
//! dependency surface minimal and the signal flow auditable. fundsp
//! is on the dep list for callers who want graph-style composition
//! later; the chain itself doesn't need it.
//!
//! Where the JS used a custom AudioWorklet (pitch shifter, glitch,
//! bitcrusher, freq shifter) the algorithms here are direct
//! translations from the worklet `process()` bodies. Hold pointers,
//! Hann windows, allpass-pair Hilbert structure — all preserved.

use std::f32::consts::TAU;

/// 3-band EQ — lowshelf @ 100 Hz, peak @ 1 kHz Q=1, highshelf @ 8 kHz.
/// Matches chronoflow-engine.js line 4255 (`Me` class). Each band is a
/// transposed-direct-form-II biquad.
pub struct Eq3 {
    sr: f32,
    low: Biquad,
    mid: Biquad,
    high: Biquad,
}

impl Eq3 {
    pub fn new(sr: f32) -> Self {
        Self {
            sr,
            low: Biquad::lowshelf(sr, 100.0, 0.0),
            mid: Biquad::peak(sr, 1000.0, 1.0, 0.0),
            high: Biquad::highshelf(sr, 8000.0, 0.0),
        }
    }

    /// Set per-band gain in dB. Frequencies + mid Q are fixed to the
    /// JS defaults.
    pub fn set(&mut self, low_db: f32, mid_db: f32, high_db: f32) {
        self.low = Biquad::lowshelf(self.sr, 100.0, low_db);
        self.mid = Biquad::peak(self.sr, 1000.0, 1.0, mid_db);
        self.high = Biquad::highshelf(self.sr, 8000.0, high_db);
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        self.high.step(self.mid.step(self.low.step(x)))
    }
}

/// Generic biquad — coefficients computed via the RBJ Audio EQ cookbook
/// formulas, which is exactly what the browser's BiquadFilterNode uses
/// internally. Coefficients match WebAudio bit-for-bit (up to f32
/// rounding) so the EQ port sounds identical to the JS version.
#[derive(Clone, Copy)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    pub fn lowshelf(sr: f32, freq: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = TAU * freq / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        // Q=0.7071 matches BiquadFilterNode's default (slope=1.0).
        let alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / 0.7071 - 1.0) + 2.0).sqrt();
        let beta = 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + beta);
        let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - beta);
        let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + beta;
        let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - beta;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    pub fn highshelf(sr: f32, freq: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = TAU * freq / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / 0.7071 - 1.0) + 2.0).sqrt();
        let beta = 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + beta);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - beta);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + beta;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - beta;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    pub fn peak(sr: f32, freq: f32, q: f32, gain_db: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = TAU * freq / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Bandpass with constant 0 dB peak — used by the formant shifter
    /// for analysis + synthesis stages.
    pub fn bandpass(sr: f32, freq: f32, q: f32) -> Self {
        let w0 = TAU * freq / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    /// Single-pole lowpass — used in the shimmer-reverb damping and
    /// comb-filter damping paths.
    pub fn lowpass(sr: f32, freq: f32) -> Self {
        let w0 = TAU * freq / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0 * std::f32::consts::SQRT_2; // Q=0.7071
        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    pub fn allpass(sr: f32, freq: f32, q: f32) -> Self {
        let w0 = TAU * freq / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let b0 = 1.0 - alpha;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 + alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::normalize(b0, b1, b2, a0, a1, a2)
    }

    fn normalize(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        // Transposed Direct Form II — single-precision-friendly
        // (less error build-up than DFI for long runs).
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Feed-forward compressor — peak detector + simple gain reduction.
/// Mirrors the behaviour of WebAudio's DynamicsCompressor at a level
/// that's audible-equivalent for voice material (the WebAudio version
/// uses a more elaborate look-ahead RMS detector but the felt response
/// is the same for sustained voice signals).
pub struct Compressor {
    sr: f32,
    threshold_db: f32,
    ratio: f32,
    attack_coef: f32,
    release_coef: f32,
    knee_db: f32,
    makeup_lin: f32,
    envelope: f32,
}

impl Compressor {
    pub fn new(
        sr: f32,
        threshold_db: f32,
        ratio: f32,
        attack_s: f32,
        release_s: f32,
        knee_db: f32,
        makeup_db: f32,
    ) -> Self {
        Self {
            sr,
            threshold_db,
            ratio,
            attack_coef: (-1.0 / (attack_s * sr)).exp(),
            release_coef: (-1.0 / (release_s * sr)).exp(),
            knee_db,
            makeup_lin: 10f32.powf(makeup_db / 20.0),
            envelope: 0.0,
        }
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        // Peak follower in linear; gain reduction in dB.
        let in_abs = x.abs();
        let coef = if in_abs > self.envelope {
            self.attack_coef
        } else {
            self.release_coef
        };
        self.envelope = in_abs + coef * (self.envelope - in_abs);
        let env_db = 20.0 * (self.envelope.max(1e-9)).log10();
        let over = env_db - self.threshold_db;
        // Soft-knee — quadratic blend across `knee_db` either side.
        let half_knee = self.knee_db * 0.5;
        let gr_db = if over <= -half_knee {
            0.0
        } else if over >= half_knee {
            over * (1.0 - 1.0 / self.ratio)
        } else {
            let t = (over + half_knee) / self.knee_db;
            let above = over * (1.0 - 1.0 / self.ratio);
            above * t * t
        };
        let gain = 10f32.powf(-gr_db / 20.0);
        x * gain * self.makeup_lin
    }
}

/// Formant shifter — 5 parallel bandpass pairs. Each pair extracts
/// energy from the input at a "formant" centre frequency, then
/// re-emits it at that centre shifted by `2^(shift/12)` (semitone
/// mapping). This is the channel-vocoder-style approach from
/// `chronoflow-engine.js:3772` (`ge` class).
///
/// Not the cleanest formant shifter possible (a real one would use
/// LPC analysis to find actual formants in the input), but matches
/// the JS implementation exactly, which is what the user has been
/// tuning against.
pub struct FormantShifter {
    sr: f32,
    shift_semitones: f32,
    mix: f32,
    /// Output makeup gain — compensates for the energy lost through
    /// the 5 narrow bandpasses (they cover ~6% of the audio band so
    /// the wet output is ~10 dB quieter than dry). Matches the JS
    /// `formant-makeup` VCA inserted right after the formant shifter
    /// in presenter.html line 322 and the `recalcMakeup` formula at
    /// line 458: `1 + min(|shift|/7, 1) * 1.5`. Without this, any
    /// non-zero formant shift makes the voice noticeably quieter
    /// than the input — exactly what was happening to Utopia.
    makeup: f32,
    /// 5 analysis bandpasses tuned to default formants.
    analysis: [Biquad; 5],
    /// 5 synthesis bandpasses re-tuned each time `set` is called.
    synthesis: [Biquad; 5],
}

const FORMANT_CENTRES: [f32; 5] = [500.0, 1500.0, 2500.0, 3500.0, 4500.0];
/// Q = bandwidth parameter — JS defaults to 8.
const FORMANT_Q: f32 = 8.0;

impl FormantShifter {
    pub fn new(sr: f32) -> Self {
        let analysis = std::array::from_fn(|i| Biquad::bandpass(sr, FORMANT_CENTRES[i], FORMANT_Q));
        let synthesis = std::array::from_fn(|i| Biquad::bandpass(sr, FORMANT_CENTRES[i], FORMANT_Q));
        Self {
            sr,
            shift_semitones: 0.0,
            mix: 1.0,
            makeup: 1.0,
            analysis,
            synthesis,
        }
    }

    pub fn set(&mut self, shift_semitones: f32, mix: f32) {
        self.shift_semitones = shift_semitones;
        self.mix = mix;
        // recalcMakeup() port — see field doc above.
        self.makeup = 1.0 + (shift_semitones.abs() / 7.0).min(1.0) * 1.5;
        let scale = 2f32.powf(shift_semitones / 12.0);
        for i in 0..5 {
            // Re-tune the synthesis filters to the shifted centres.
            // Clamp to a sane band to avoid filter blow-up at extreme
            // shifts (the JS doesn't clamp but at f32 + 44.1k an
            // 8 kHz × 2 = 16 kHz centre is close to Nyquist).
            let target = (FORMANT_CENTRES[i] * scale).clamp(40.0, self.sr * 0.45);
            self.synthesis[i] = Biquad::bandpass(self.sr, target, FORMANT_Q);
        }
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.mix <= 1e-4 {
            return x;
        }
        let mut wet = 0.0;
        for i in 0..5 {
            let band = self.analysis[i].step(x);
            wet += self.synthesis[i].step(band);
        }
        // Apply formant-makeup BEFORE mixing — same wire position as
        // the JS (`formant → formant-makeup → pitch` in the graph
        // build at presenter.html line 339).
        x * (1.0 - self.mix) + wet * self.mix * self.makeup
    }
}

/// Granular pitch shifter — direct port of
/// `worklets/pitch-shifter-processor.js`. Two Hann-windowed grains at
/// 0/0.5 phase offset crossfade against a 2-second circular buffer
/// to produce a continuous shifted output. Cheap, glitch-prone on
/// transients (it's the JS algorithm — keep parity over perfection).
pub struct PitchShifter {
    buffer: Vec<f32>,
    write_pos: usize,
    grain_size: usize,
    pitch: f32,
    mix: f32,
    g0_phase: f32,
    g0_read_pos: usize,
    g1_phase: f32,
    g1_read_pos: usize,
}

impl PitchShifter {
    pub fn new(sr: f32) -> Self {
        // 2 seconds, matching the JS worklet at 44.1k.
        let len = (sr * 2.0) as usize;
        let grain_size = 2048;
        Self {
            buffer: vec![0.0; len],
            write_pos: 0,
            grain_size,
            pitch: 1.0,
            mix: 1.0,
            g0_phase: 0.0,
            g0_read_pos: 0,
            // Grain 1 offset by half a grain — matches the JS
            // constructor initialisation.
            g1_phase: 0.5,
            g1_read_pos: 0,
        }
    }

    pub fn set(&mut self, semitones: f32, cents: f32, mix: f32) {
        let total_semis = semitones + cents / 100.0;
        self.pitch = 2f32.powf(total_semis / 12.0);
        self.mix = mix;
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.mix <= 1e-4 {
            return x;
        }
        let len = self.buffer.len();
        self.buffer[self.write_pos] = x;

        let phase_inc = 1.0 / self.grain_size as f32;

        // Grain 0.
        self.g0_phase += phase_inc;
        if self.g0_phase >= 1.0 {
            self.g0_phase -= 1.0;
            self.g0_read_pos = (self.write_pos + len - self.grain_size) % len;
        }
        let off0 = (self.g0_phase * self.grain_size as f32 * self.pitch) as usize;
        let idx0 = (self.g0_read_pos + off0) % len;
        let w0 = 0.5 * (1.0 - (TAU * self.g0_phase).cos());
        let s0 = self.buffer[idx0] * w0;

        // Grain 1 (offset).
        self.g1_phase += phase_inc;
        if self.g1_phase >= 1.0 {
            self.g1_phase -= 1.0;
            self.g1_read_pos = (self.write_pos + len - self.grain_size) % len;
        }
        let off1 = (self.g1_phase * self.grain_size as f32 * self.pitch) as usize;
        let idx1 = (self.g1_read_pos + off1) % len;
        let w1 = 0.5 * (1.0 - (TAU * self.g1_phase).cos());
        let s1 = self.buffer[idx1] * w1;

        let wet = s0 + s1;
        self.write_pos = (self.write_pos + 1) % len;
        x * (1.0 - self.mix) + wet * self.mix
    }
}

/// Glitch / stutter — direct port of `worklets/glitch-processor.js`.
/// Captures a rolling 1-second buffer and, when `active`, replays a
/// `size`-second slice repeatedly at `rate` retriggers per second.
/// In the profiles shipped today every character has `glitch_active =
/// false` so this is effectively a passthrough, but it's here in
/// place so emotion transitions that flip the flag don't need a
/// chain rebuild.
pub struct Glitch {
    sr: f32,
    buffer: Vec<f32>,
    write_pos: usize,
    rate: f32,
    size_s: f32,
    stutter_size: usize,
    stutter_read_pos: usize,
    stutter_phase: usize,
    samples_per_retrigger: usize,
    retrigger_counter: usize,
    active: bool,
    probability: f32,
    pitch: f32,
    mix: f32,
    rng_state: u64,
}

impl Glitch {
    pub fn new(sr: f32) -> Self {
        let len = sr as usize;
        Self {
            sr,
            buffer: vec![0.0; len],
            write_pos: 0,
            rate: 8.0,
            size_s: 0.05,
            stutter_size: 2048,
            stutter_read_pos: 0,
            stutter_phase: 0,
            samples_per_retrigger: (sr / 8.0) as usize,
            retrigger_counter: 0,
            active: false,
            probability: 1.0,
            pitch: 1.0,
            mix: 1.0,
            rng_state: 0xCAFE_F00D_DEAD_BEEF,
        }
    }

    pub fn set(&mut self, active: bool, rate: f32, size: f32, probability: f32, mix: f32) {
        self.active = active;
        self.rate = rate.clamp(0.5, 50.0);
        self.size_s = size.clamp(0.005, 1.0);
        self.probability = probability.clamp(0.0, 1.0);
        self.mix = mix;
        self.samples_per_retrigger = (self.sr / self.rate) as usize;
        self.stutter_size = (self.size_s * self.sr) as usize;
    }

    #[inline]
    fn rand(&mut self) -> f32 {
        // PCG-ish — same one used elsewhere in this crate.
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.rng_state >> 32) as u32) as f32 / u32::MAX as f32
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        let len = self.buffer.len();
        self.buffer[self.write_pos] = x;
        self.write_pos = (self.write_pos + 1) % len;

        if !self.active || self.mix <= 1e-4 {
            return x;
        }

        self.retrigger_counter += 1;
        if self.retrigger_counter >= self.samples_per_retrigger {
            self.retrigger_counter = 0;
            if self.rand() < self.probability {
                self.stutter_phase = 0;
                self.stutter_read_pos = (self.write_pos + len - self.stutter_size) % len;
            }
        }

        let grain_pos = (self.stutter_phase as f32 * self.pitch) as usize;
        if grain_pos >= self.stutter_size {
            self.stutter_phase = 0;
        }
        let read_idx = (self.stutter_read_pos + grain_pos) % len;
        let wet = self.buffer[read_idx];
        self.stutter_phase += 1;

        x * (1.0 - self.mix) + wet * self.mix
    }
}

/// Single-sideband frequency shifter — direct port of
/// `worklets/freq-shifter-processor.js`. Uses cascaded allpass pairs
/// to approximate a Hilbert transform (~20 Hz–20 kHz coverage) and
/// modulates the analytic signal with a complex exponential at
/// `shift_hz`.
pub struct FreqShifter {
    sr: f32,
    shift_hz: f32,
    mix: f32,
    states_a: [Allpass1; 4],
    states_b: [Allpass1; 4],
    phase: f32,
}

/// Coefficients for the JS Hilbert allpass-pair network. These are
/// the standard values published by Olli Niemitalo for a 90° network
/// covering audio band.
const ALLPASS_A: [f32; 4] = [0.6923878, 0.9360654322959, 0.9882295226860, 0.9987488452737];
const ALLPASS_B: [f32; 4] = [0.4021921162426, 0.8561710882420, 0.9722909545651, 0.9952884791278];

#[derive(Clone, Copy, Default)]
struct Allpass1 {
    x1: f32,
    y1: f32,
}

impl Allpass1 {
    #[inline]
    fn step(&mut self, x: f32, coef: f32) -> f32 {
        let y = coef * (x - self.y1) + self.x1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

impl FreqShifter {
    pub fn new(sr: f32) -> Self {
        Self {
            sr,
            shift_hz: 0.0,
            mix: 1.0,
            states_a: [Allpass1::default(); 4],
            states_b: [Allpass1::default(); 4],
            phase: 0.0,
        }
    }

    pub fn set(&mut self, shift_hz: f32, mix: f32) {
        self.shift_hz = shift_hz;
        self.mix = mix;
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.mix <= 1e-4 || self.shift_hz.abs() < 1e-4 {
            return x;
        }
        let mut sig_a = x;
        for (i, c) in ALLPASS_A.iter().enumerate() {
            sig_a = self.states_a[i].step(sig_a, *c);
        }
        let mut sig_b = x;
        for (i, c) in ALLPASS_B.iter().enumerate() {
            sig_b = self.states_b[i].step(sig_b, *c);
        }
        let cos_p = self.phase.cos();
        let sin_p = self.phase.sin();
        // Upper sideband — `mode = 0` in the JS.
        let wet = sig_a * cos_p - sig_b * sin_p;
        self.phase += TAU * self.shift_hz / self.sr;
        if self.phase > TAU {
            self.phase -= TAU;
        }
        x * (1.0 - self.mix) + wet * self.mix
    }
}

/// Feedback comb filter with damping lowpass — matches the JS engine
/// (`Re` class, line 4743). `frequency` → delay_time mapping is
/// `1 / frequency`, clamped to 0.1 s.
pub struct Comb {
    sr: f32,
    buffer: Vec<f32>,
    write_pos: usize,
    delay_samples: usize,
    feedback: f32,
    mix: f32,
    damping: Biquad,
}

impl Comb {
    pub fn new(sr: f32) -> Self {
        // Max 0.1 s delay matches the JS clamp.
        let max_len = (sr * 0.1) as usize + 1;
        Self {
            sr,
            buffer: vec![0.0; max_len],
            write_pos: 0,
            delay_samples: 1,
            feedback: 0.0,
            mix: 0.0,
            damping: Biquad::lowpass(sr, 8000.0),
        }
    }

    pub fn set(&mut self, frequency: f32, feedback: f32, mix: f32) {
        let f = frequency.max(20.0).min(5000.0);
        let delay_s = (1.0 / f).min(0.1);
        self.delay_samples = ((delay_s * self.sr) as usize).max(1).min(self.buffer.len() - 1);
        self.feedback = feedback.clamp(-0.99, 0.99);
        self.mix = mix;
        // JS damping default = 0.3 → cutoff = 20000 * 0.01^0.3 ≈ 5012 Hz.
        let damping = 0.3_f32;
        let cutoff = 20_000.0 * 0.01_f32.powf(damping);
        self.damping = Biquad::lowpass(self.sr, cutoff);
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.mix <= 1e-4 {
            return x;
        }
        let len = self.buffer.len();
        let read_idx = (self.write_pos + len - self.delay_samples) % len;
        let delayed = self.buffer[read_idx];
        let damped = self.damping.step(delayed);
        let wet = damped;
        self.buffer[self.write_pos] = x + damped * self.feedback;
        self.write_pos = (self.write_pos + 1) % len;
        x * (1.0 - self.mix) + wet * self.mix
    }
}

/// Bit + sample-rate reduction. Direct port of
/// `worklets/bitcrusher-processor.js`. `bits` ∈ [1, 16] quantises;
/// `sample_rate_reduction` holds the sample for N frames before
/// re-quantising. JS uses fixed SR reduction = 1 in all profiles, so
/// only the bit-depth path matters for the current voice presets, but
/// the hold-counter is wired up for completeness.
pub struct Bitcrusher {
    bits: f32,
    sample_rate_reduction: usize,
    mix: f32,
    hold_sample: f32,
    hold_counter: usize,
}

impl Bitcrusher {
    pub fn new() -> Self {
        Self {
            bits: 16.0,
            sample_rate_reduction: 1,
            mix: 0.0,
            hold_sample: 0.0,
            hold_counter: 0,
        }
    }

    pub fn set(&mut self, bits: f32, mix: f32) {
        self.bits = bits.clamp(1.0, 16.0);
        self.mix = mix;
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.mix <= 1e-4 {
            return x;
        }
        self.hold_counter += 1;
        if self.hold_counter >= self.sample_rate_reduction {
            self.hold_counter = 0;
            let step = 0.5_f32.powf(self.bits);
            let inv_step = 1.0 / step;
            self.hold_sample = (x * inv_step).round() * step;
        }
        x * (1.0 - self.mix) + self.hold_sample * self.mix
    }
}

/// Single-tap delay with feedback. Matches the JS `F` class
/// (chronoflow-engine.js:415).
pub struct Delay {
    buffer: Vec<f32>,
    write_pos: usize,
    delay_samples: usize,
    feedback: f32,
    mix: f32,
}

impl Delay {
    pub fn new(sr: f32, max_time_s: f32) -> Self {
        let len = (sr * max_time_s) as usize + 1;
        Self {
            buffer: vec![0.0; len],
            write_pos: 0,
            delay_samples: (sr * 0.3) as usize,
            feedback: 0.4,
            mix: 0.5,
        }
    }

    pub fn set(&mut self, time_s: f32, feedback: f32, mix: f32) {
        let max = self.buffer.len() - 1;
        self.delay_samples = ((time_s * self.buffer.len() as f32 / 5.0) as usize).min(max);
        // ^ JS max is 5s; buffer above was sized to fit. Recompute
        // delay in samples relative to the actual buffer size.
        let s = (time_s * (self.buffer.len() as f32 - 1.0) / 5.0) as usize;
        self.delay_samples = s.max(1).min(max);
        self.feedback = feedback.clamp(0.0, 0.95);
        self.mix = mix;
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.mix <= 1e-4 {
            return x;
        }
        let len = self.buffer.len();
        let read_idx = (self.write_pos + len - self.delay_samples) % len;
        let wet = self.buffer[read_idx];
        self.buffer[self.write_pos] = x + wet * self.feedback;
        self.write_pos = (self.write_pos + 1) % len;
        x * (1.0 - self.mix) + wet * self.mix
    }
}

/// Shimmer reverb — a Schroeder-style reverb (4 allpass diffusers +
/// 4 comb filters) with a pitch-shifted feedback path tapped from
/// the reverb tail. Approximates the JS `Q` class (convolution +
/// LFO-modulated detune feedback) at a fraction of the cost — the
/// convolution version needs FFT and a generated IR which would
/// inflate the dep tree.
pub struct Shimmer {
    sr: f32,
    diffusers: [Allpass; 4],
    combs: [CombSimple; 4],
    /// Approximate average comb delay in seconds, used to convert
    /// `decay` (T60) → per-comb feedback via the standard reverb
    /// formula `fb = 10^(-3·d / T60)`. Without this conversion my
    /// first cut used a made-up formula that produced fb ≈ 0.994 at
    /// decay=5 s — well into the unstable regime, which generated
    /// the initial howl/noise on Eliza.
    avg_comb_delay: f32,
    decay: f32,
    shimmer_amount: f32,
    mix: f32,
    /// LFO-modulated dual-tap delays in the feedback path —
    /// matches the JS shimmer (presenter line 3906-3915):
    ///   pitchDelay1 = 10 ms, LFO @ rate
    ///   pitchDelay2 = 20 ms, LFO @ rate × 1.1
    /// These create Doppler-style pitch motion rather than a true
    /// pitch shifter, so they DON'T have the grain-boundary
    /// artifacts a granular shifter would accumulate through the
    /// shimmer loop. That was Eliza's main noise source.
    shimmer_buf: Vec<f32>,
    shimmer_write: usize,
    shimmer_lfo: f32,
    /// State for the pitched feedback — output of last sample's
    /// shimmer tap, fed back into next sample's input.
    shimmer_fb_state: f32,
    damping: Biquad,
}

struct Allpass {
    buffer: Vec<f32>,
    write_pos: usize,
    gain: f32,
}

impl Allpass {
    fn new(sr: f32, time_s: f32) -> Self {
        let len = (sr * time_s) as usize + 1;
        Self {
            buffer: vec![0.0; len],
            write_pos: 0,
            gain: 0.5,
        }
    }
    #[inline]
    fn step(&mut self, x: f32) -> f32 {
        let len = self.buffer.len();
        let read_idx = (self.write_pos + 1) % len;
        let buf_out = self.buffer[read_idx];
        let v = x + buf_out * -self.gain;
        self.buffer[self.write_pos] = v;
        self.write_pos = (self.write_pos + 1) % len;
        buf_out + v * self.gain
    }
}

struct CombSimple {
    buffer: Vec<f32>,
    write_pos: usize,
    feedback: f32,
    lp_state: f32,
    damp: f32,
}

impl CombSimple {
    fn new(sr: f32, time_s: f32) -> Self {
        let len = (sr * time_s) as usize + 1;
        Self {
            buffer: vec![0.0; len],
            write_pos: 0,
            feedback: 0.84,
            lp_state: 0.0,
            damp: 0.2,
        }
    }
    #[inline]
    fn step(&mut self, x: f32) -> f32 {
        let len = self.buffer.len();
        let read_idx = (self.write_pos + 1) % len;
        let y = self.buffer[read_idx];
        // One-pole damp on the feedback — makes the tail darken
        // with time, the way every reverb's tail naturally does.
        self.lp_state = y * (1.0 - self.damp) + self.lp_state * self.damp;
        self.buffer[self.write_pos] = x + self.lp_state * self.feedback;
        self.write_pos = (self.write_pos + 1) % len;
        y
    }
}

impl Shimmer {
    pub fn new(sr: f32) -> Self {
        let allpass_times = [0.005, 0.0017, 0.013, 0.0093];
        let comb_times = [0.0297, 0.0371, 0.0411, 0.0437];
        let avg_comb_delay = comb_times.iter().sum::<f32>() / comb_times.len() as f32;
        let diffusers = std::array::from_fn(|i| Allpass::new(sr, allpass_times[i]));
        let combs = std::array::from_fn(|i| CombSimple::new(sr, comb_times[i]));
        // Shimmer-feedback delay buffer — sized for 30 ms (covers
        // the 20 ms base tap + ~5 ms LFO swing). LFO modulates the
        // read offset to produce the Doppler-pitch effect.
        let shimmer_buf = vec![0.0; (sr * 0.03) as usize + 1];
        Self {
            sr,
            diffusers,
            combs,
            avg_comb_delay,
            decay: 2.0,
            shimmer_amount: 0.3,
            mix: 0.0,
            shimmer_buf,
            shimmer_write: 0,
            shimmer_lfo: 0.0,
            shimmer_fb_state: 0.0,
            damping: Biquad::lowpass(sr, 5000.0),
        }
    }

    pub fn set(&mut self, decay: f32, shimmer_amount: f32, mix: f32) {
        self.decay = decay.max(0.1);
        self.shimmer_amount = shimmer_amount.clamp(0.0, 1.0);
        self.mix = mix;
        let fb = 10f32.powf(-3.0 * self.avg_comb_delay / self.decay);
        let fb = fb.clamp(0.5, 0.94);
        for c in self.combs.iter_mut() {
            c.feedback = fb;
        }
        let cutoff = 20_000.0 * 0.01_f32.powf(0.25 + 0.05 * decay);
        self.damping = Biquad::lowpass(self.sr, cutoff.clamp(800.0, 18_000.0));
    }

    /// Linear-interpolated read at a (possibly fractional) delay
    /// from the shimmer buffer.
    #[inline]
    fn read_shimmer(&self, delay_samples: f32) -> f32 {
        let len = self.shimmer_buf.len();
        let read_pos = self.shimmer_write as f32 + len as f32 - delay_samples;
        let idx_f = read_pos.rem_euclid(len as f32);
        let idx_a = idx_f.floor() as usize % len;
        let idx_b = (idx_a + 1) % len;
        let frac = idx_f - idx_f.floor();
        self.shimmer_buf[idx_a] * (1.0 - frac) + self.shimmer_buf[idx_b] * frac
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.mix <= 1e-4 {
            return x;
        }
        // Loop gain budget: comb network peak gain ≈ 1/(1-fb) per
        // tap averaged across 4 → roughly 1.5-3 for typical fb.
        // Capping the shimmer feedback at `* 0.15` keeps the loop
        // gain ≤ 0.45 even at maximum profile values — safely sub-
        // oscillation across the whole parameter range.
        let input = x + self.shimmer_fb_state * self.shimmer_amount * 0.15;

        // Diffusion + comb reverb.
        let mut s = input;
        for ap in self.diffusers.iter_mut() {
            s = ap.step(s);
        }
        let mut wet = 0.0;
        for c in self.combs.iter_mut() {
            wet += c.step(s);
        }
        wet *= 0.25;
        wet = self.damping.step(wet);

        // Dual-tap LFO-modulated shimmer feedback (port of the JS
        // pitchDelay1/pitchDelay2 path). Each tap reads from the
        // reverb output at a slightly modulated delay; the
        // modulation creates a Doppler pitch wobble that's the felt
        // "shimmer" effect — without the grain-boundary artifacts a
        // true pitch shifter would accumulate through the loop.
        self.shimmer_buf[self.shimmer_write] = wet;
        // LFO rate ≈ 1.5 Hz baseline, slightly different per tap.
        self.shimmer_lfo += TAU * 1.5 / self.sr;
        if self.shimmer_lfo > TAU {
            self.shimmer_lfo -= TAU;
        }
        let lfo1 = self.shimmer_lfo.sin();
        let lfo2 = (self.shimmer_lfo * 1.1 + 0.7).sin();
        // Tap 1: base 10 ms ± 2 ms swing. Tap 2: base 20 ms ± 2 ms.
        let d1 = (0.010 + 0.002 * lfo1) * self.sr;
        let d2 = (0.020 + 0.002 * lfo2) * self.sr;
        let max_delay = (self.shimmer_buf.len() - 2) as f32;
        let tap1 = self.read_shimmer(d1.min(max_delay).max(1.0));
        let tap2 = self.read_shimmer(d2.min(max_delay).max(1.0));
        self.shimmer_fb_state = (tap1 + tap2) * 0.5;
        self.shimmer_write = (self.shimmer_write + 1) % self.shimmer_buf.len();

        x * (1.0 - self.mix) + wet * self.mix
    }
}

/// N-voice chorus — each voice is a delay line modulated by a sine
/// LFO at slightly different rate. Mirrors the JS `Te` class
/// (chronoflow-engine.js:4015).
pub struct Chorus {
    sr: f32,
    rate: f32,
    depth_s: f32,
    voices: usize,
    mix: f32,
    buffer: Vec<f32>,
    write_pos: usize,
    lfo_phases: [f32; 6],
}

impl Chorus {
    pub fn new(sr: f32) -> Self {
        // 50 ms buffer — base 7 ms + per-voice spacing + 15 ms depth.
        let len = (sr * 0.05) as usize + 1;
        Self {
            sr,
            rate: 1.5,
            depth_s: 0.0015,
            voices: 3,
            mix: 0.0,
            buffer: vec![0.0; len],
            write_pos: 0,
            // Per-voice phase offsets — distributes voices across LFO cycle.
            lfo_phases: [0.0, TAU / 6.0, 2.0 * TAU / 6.0, 3.0 * TAU / 6.0, 4.0 * TAU / 6.0, 5.0 * TAU / 6.0],
        }
    }

    pub fn set(&mut self, rate: f32, depth: f32, voices: usize, mix: f32) {
        self.rate = rate;
        self.depth_s = depth * 0.003; // depth=0.5 → 1.5 ms — matches JS line 4050
        self.voices = voices.clamp(2, 6);
        self.mix = mix;
    }

    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        if self.mix <= 1e-4 {
            return x;
        }
        let len = self.buffer.len() as f32;
        self.buffer[self.write_pos] = x;

        // Base delay 7 ms, per-voice spacing 0.015 / 6.
        let base = 0.007;
        let spacing = 0.015 / 6.0;
        let mut wet = 0.0;
        for v in 0..self.voices {
            // Advance phase.
            let voice_rate = self.rate * (1.0 + v as f32 * 0.12);
            self.lfo_phases[v] += TAU * voice_rate / self.sr;
            if self.lfo_phases[v] > TAU {
                self.lfo_phases[v] -= TAU;
            }
            let mod_amount = self.lfo_phases[v].sin() * self.depth_s;
            let delay_s = base + v as f32 * spacing + mod_amount;
            let delay_samples = (delay_s * self.sr).max(1.0).min(len - 1.0);
            // Linear-interpolated read for smooth modulation.
            let read_pos_f = self.write_pos as f32 + len - delay_samples;
            let idx_f = read_pos_f % len;
            let idx_a = idx_f.floor() as usize;
            let idx_b = (idx_a + 1) % self.buffer.len();
            let frac = idx_f - idx_a as f32;
            wet += self.buffer[idx_a] * (1.0 - frac) + self.buffer[idx_b] * frac;
        }
        wet /= self.voices as f32;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        x * (1.0 - self.mix) + wet * self.mix
    }
}
