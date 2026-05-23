//! Render a face for one frame.
//!
//! [`Renderer`] is a soft pipeline: software-rasterizes a particle
//! field + contours into a `tiny_skia::Pixmap`. Cheap enough to run at
//! 30fps on a single core at 640×360 with ~12K particles; pluggable
//! into a video encoder downstream (Eliza's MoQ path uses the same
//! Pixmap shape).
//!
//! Animation state lives in [`FaceState`] — separated from the
//! immutable [`Character`] so a single character can drive many tiles
//! and a single tile can swap characters without rebuilding particles.

use tiny_skia::{Color, Paint, PathBuilder, Pixmap, Stroke, Transform};

use crate::character::{AccentParticles, Character, ContourBaseline, EmberConfig, NebulaCloud};
use crate::face::{generate_face, Particle};

/// Linear interpolation between two bytes — returns a `u8` at `t=0.5`
/// midway between `a` and `b`. `t` is clamped to `0..=1`.
#[inline]
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let af = a as f32;
    let bf = b as f32;
    (af + (bf - af) * t.clamp(0.0, 1.0)) as u8
}

/// Step a PCG-like u64 RNG and return a `0..1` float. Tiny helper for
/// renderer-thread randomness — no `rand` dependency, no thread-local
/// thrash, deterministic from a seed.
#[inline]
fn next_rand(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 32) as u32) as f32 / u32::MAX as f32
}

/// A single live ember — a spark drifting upward off the face.
#[derive(Clone, Copy, Debug)]
struct Ember {
    /// Position in scene space (post-scale, pre-projection — same
    /// frame the particle field lives in).
    pos: [f32; 3],
    /// Velocity. Mostly +y (rising) with a small per-ember x wander.
    vel: [f32; 3],
    /// Seconds since spawn.
    age: f32,
    /// Per-ember seed used for horizontal flutter + size jitter.
    seed: f32,
}

/// Per-tile mutable state. Holds the particle field plus the cached
/// animation clock — separate from [`Character`] so the same character
/// can be rendered on N independent tiles.
pub struct FaceState {
    pub character_name: &'static str,
    particles: Vec<Particle>,
    /// Live position per particle (eases from `scattered` toward
    /// `target` as `materialized` rises). Same length as `particles`.
    live: Vec<[f32; 3]>,
    /// Where each particle goes when scattered (set by the transition).
    scattered: Vec<[f32; 3]>,
    /// `0..=1` materialization per particle.
    pt: Vec<f32>,
    /// Global `0..=1`: 0 = fully scattered, 1 = fully on the face.
    pub materialized: f32,
    /// `0..=1` audio loudness — written by the host each frame.
    /// Drives a per-particle reactive jitter so the face *visibly*
    /// shimmers when the agent (or someone else) is speaking. Without
    /// this the face renders motionless except for the breathing
    /// pulse, which reads as a still image.
    pub audio_level: f32,
    /// Current head yaw + pitch (radians). Eases each frame toward
    /// `gaze_yaw_target` / `gaze_pitch_target` — see [`Self::step_gaze`].
    pub gaze_yaw: f32,
    pub gaze_pitch: f32,
    pub gaze_yaw_target: f32,
    pub gaze_pitch_target: f32,
    /// When the next gaze shift fires. Picked uniformly in a window
    /// each time a shift lands, so the head moves at uneven natural
    /// intervals rather than on a fixed clock.
    next_gaze_shift_at: f32,
    /// Simple PRNG state for gaze picks — owned by the state so the
    /// `step_gaze` call doesn't need a thread RNG (renderer thread is
    /// not async-runtime-friendly for `rand`).
    gaze_rng: u64,
    /// Live embers (when the character configured an `EmberConfig`).
    /// Renderer ticks this every frame: ages each ember, removes the
    /// expired, spawns new ones up to `EmberConfig::max_alive`.
    embers: Vec<Ember>,
    /// Fractional spawn budget — accumulates `spawn_rate * dt` each
    /// frame so we can spawn N integer embers when the budget rolls
    /// over. Without this a `spawn_rate=70` with `dt=1/15` would
    /// always round to 4 and we'd never see the 5th.
    ember_spawn_budget: f32,
    /// PRNG for ember spawning.
    ember_rng: u64,
    /// `0..=1` blink amount — 0 = eyes open, 1 = fully closed. Eased
    /// up/down on a per-blink cycle. Renderer reads this and dims
    /// eye-particle alpha by it.
    blink: f32,
    /// Blink direction: `+1.0` while closing (blink rising), `-1.0`
    /// while opening (blink falling), `0.0` between blinks.
    blink_dir: f32,
    /// When the next blink starts (in `time` seconds).
    next_blink_at: f32,
    /// PRNG for blink scheduling.
    blink_rng: u64,
    /// Eye saccade — small offset on top of head yaw/pitch. Eyes
    /// dart slightly independent of the head for a more alive feel.
    /// Renderer adds (saccade_x, saccade_y) to eye-particle screen
    /// position.
    eye_saccade_x: f32,
    eye_saccade_y: f32,
    eye_saccade_target_x: f32,
    eye_saccade_target_y: f32,
    next_saccade_at: f32,
    saccade_rng: u64,
    /// Speech-onset flash — set to `time` when the audio level crosses
    /// a rising threshold. Renderer fades a full-field bright tint
    /// over `SPEECH_FLASH_DURATION` after this. Reads as her
    /// *reacting* to her own speech rather than passively pulsing.
    speech_onset_at: f32,
    /// Previous audio level — used by [`step_audio_onset`] to detect
    /// the rising edge.
    prev_audio_level: f32,
    /// Brow expression — `-1` (full furrow, eyes lowered) to `+1`
    /// (full raise, surprise). Renderer offsets brow-particle Y by
    /// this. Host sets it via [`set_brow`] based on emotion.
    brow: f32,
    /// Camera shake amount in pixels — eases down each frame.
    /// [`step_audio_onset`] pumps it up when audio peaks; the
    /// renderer translates the whole particle field by a sub-pixel
    /// jitter scaled by this value. Reads as the floor shaking.
    shake: f32,
    /// Phase for camera-shake jitter — drives the deterministic
    /// random walk that turns `shake` into per-frame x/y offsets.
    shake_phase: f32,
}

impl FaceState {
    /// Build a fresh particle field for `character`. `particle_count`
    /// is total field size — about 60% of the avatar's count looks
    /// great at 640×360.
    pub fn new(character: &Character, particle_count: usize, scale: f32, seed: u64) -> Self {
        let particles = generate_face(particle_count, scale, &character.geometry, &character.palette, seed);
        let live: Vec<[f32; 3]> = particles.iter().map(|p| p.target).collect();
        // Initial scatter positions — replaced as soon as the
        // transition fires; reasonable default in case the renderer is
        // asked for a frame before scattering.
        let scattered: Vec<[f32; 3]> = particles.iter().map(|p| p.target).collect();
        let pt: Vec<f32> = vec![1.0; particles.len()]; // start materialized
        Self {
            character_name: character.name,
            particles,
            live,
            scattered,
            pt,
            materialized: 1.0,
            audio_level: 0.0,
            gaze_yaw: 0.0,
            gaze_pitch: 0.0,
            gaze_yaw_target: 0.0,
            gaze_pitch_target: 0.0,
            // First shift fires in 2-4s — quick enough to feel alive
            // from the moment she appears.
            next_gaze_shift_at: 3.0,
            gaze_rng: seed.wrapping_add(0xA5A5_5A5A_C3C3_3C3C),
            embers: Vec::new(),
            ember_spawn_budget: 0.0,
            ember_rng: seed.wrapping_add(0xDEAD_BEEF_CAFE_F00D),
            blink: 0.0,
            blink_dir: 0.0,
            // First blink fires in 2-5s.
            next_blink_at: 3.0,
            blink_rng: seed.wrapping_add(0x1234_5678_9ABC_DEF0),
            eye_saccade_x: 0.0,
            eye_saccade_y: 0.0,
            eye_saccade_target_x: 0.0,
            eye_saccade_target_y: 0.0,
            // First saccade fires quickly — gives instant aliveness.
            next_saccade_at: 1.5,
            saccade_rng: seed.wrapping_add(0xFACE_BEEF_BAAA_AAAD),
            speech_onset_at: -10.0,
            prev_audio_level: 0.0,
            brow: 0.0,
            shake: 0.0,
            shake_phase: 0.0,
        }
    }

    /// Detect audio onsets — a rising edge in `audio_level` — and:
    /// 1) trigger a brief full-field bright-tint flash, and
    /// 2) pump the camera shake.
    /// Call once per frame from the host (after `set_audio_level`).
    pub fn step_audio_onset(&mut self, time: f32, dt: f32) {
        let cur = self.audio_level;
        let prev = self.prev_audio_level;
        // Onset = a clear rising edge to a non-trivial level. Tuned
        // so a single loud syllable fires it, but the background hum
        // of a quiet conversation doesn't.
        if cur > 0.18 && cur - prev > 0.08 && time - self.speech_onset_at > 0.45 {
            self.speech_onset_at = time;
            // Shake jolt — bigger jolt for louder onset.
            self.shake = (self.shake + cur * 6.0).min(8.0);
        }
        self.prev_audio_level = cur;
        // Shake decays.
        self.shake = (self.shake - dt * 12.0).max(0.0);
        self.shake_phase += dt * 60.0;
    }

    /// Step the eye saccade — eyes drift to a new small offset every
    /// 1-3s, eased toward target so the dart is fast but not snappy.
    /// Magnitude is small (a few % of face width) — humans saccade
    /// constantly without their face moving with them.
    pub fn step_eye_saccade(&mut self, time: f32, dt: f32) {
        if time >= self.next_saccade_at {
            let r1 = next_rand(&mut self.saccade_rng);
            let r2 = next_rand(&mut self.saccade_rng);
            let r3 = next_rand(&mut self.saccade_rng);
            self.eye_saccade_target_x = (r1 - 0.5) * 0.12;
            self.eye_saccade_target_y = (r2 - 0.5) * 0.06;
            self.next_saccade_at = time + 1.0 + r3 * 2.0;
        }
        let speed = 6.0; // fast — saccades are quick
        self.eye_saccade_x +=
            (self.eye_saccade_target_x - self.eye_saccade_x) * (dt * speed).min(1.0);
        self.eye_saccade_y +=
            (self.eye_saccade_target_y - self.eye_saccade_y) * (dt * speed).min(1.0);
    }

    /// Set the brow expression. `-1.0` = full furrow (concern,
    /// concentration), `0.0` = neutral, `+1.0` = full raise
    /// (surprise, curiosity). Host typically derives from the active
    /// emotion. Clamped on read.
    pub fn set_brow(&mut self, amount: f32) {
        self.brow = amount.clamp(-1.0, 1.0);
    }

    /// Step the blink. Eyes briefly close at irregular intervals
    /// (every 3-7s) — gives an instant "this is alive" cue. The
    /// renderer reads `self.blink` (0=open, 1=closed) and dims
    /// eye-particle alpha by it. A blink lasts ~150ms: ramp up to
    /// fully closed in ~80ms, ramp back open in ~80ms.
    pub fn step_blink(&mut self, time: f32, dt: f32) {
        // Idle → schedule the next blink.
        if self.blink_dir == 0.0 && time >= self.next_blink_at {
            self.blink_dir = 1.0;
        }
        if self.blink_dir > 0.0 {
            // Closing.
            self.blink += dt * 13.0;
            if self.blink >= 1.0 {
                self.blink = 1.0;
                self.blink_dir = -1.0;
            }
        } else if self.blink_dir < 0.0 {
            // Opening.
            self.blink -= dt * 13.0;
            if self.blink <= 0.0 {
                self.blink = 0.0;
                self.blink_dir = 0.0;
                let r = next_rand(&mut self.blink_rng);
                self.next_blink_at = time + 3.0 + r * 4.0;
            }
        }
    }

    /// Advance the ember swarm one frame. Spawns new embers up to the
    /// configured `max_alive` cap, ages each one, drops expired. Call
    /// once per frame from the host (skipped automatically when the
    /// character has no `EmberConfig` — cheap conditional inside).
    pub fn step_embers(&mut self, cfg: &EmberConfig, dt: f32, scale: f32) {
        // ── Spawn ──
        self.ember_spawn_budget += cfg.spawn_rate * dt;
        while self.ember_spawn_budget >= 1.0 && self.embers.len() < cfg.max_alive {
            self.ember_spawn_budget -= 1.0;
            // PRNG step — three rolls per spawn.
            let r1 = next_rand(&mut self.ember_rng);
            let r2 = next_rand(&mut self.ember_rng);
            let r3 = next_rand(&mut self.ember_rng);
            let r4 = next_rand(&mut self.ember_rng);
            let r5 = next_rand(&mut self.ember_rng);
            let r6 = next_rand(&mut self.ember_rng);
            // Spawn point: across the lower half of the face. x ∈
            // [-0.35, 0.35], y ∈ [-0.45, 0.1], small z jitter.
            let sx = (r1 - 0.5) * 0.7 * scale;
            let sy = (-0.45 + r2 * 0.55) * scale;
            let sz = (r3 - 0.5) * 0.2 * scale;
            // Velocity: mostly upward + small horizontal wander +
            // tiny outward z.
            let vx = (r4 - 0.5) * 0.35;
            let vy = cfg.rise_speed * (0.6 + r5 * 0.8);
            let vz = (r6 - 0.5) * 0.2;
            self.embers.push(Ember {
                pos: [sx, sy, sz],
                vel: [vx, vy, vz],
                age: 0.0,
                seed: r1,
            });
        }
        // ── Age + advance + cull ──
        self.embers.retain_mut(|e| {
            e.age += dt;
            if e.age >= cfg.lifetime {
                return false;
            }
            // Sinusoidal horizontal flutter — embers don't fly
            // straight up, they curl.
            let flutter = (e.age * 4.0 + e.seed * 13.0).sin() * 0.08;
            e.pos[0] += (e.vel[0] + flutter) * dt;
            e.pos[1] += e.vel[1] * dt;
            e.pos[2] += e.vel[2] * dt;
            // Slow down vertically as they cool — drag.
            e.vel[1] *= 1.0 - dt * 0.4;
            true
        });
    }

    /// Step the head gaze. Call once per frame from the renderer with
    /// the current animation `time` and frame delta `dt`. Picks a new
    /// target every 2.5-7s, eases the current yaw/pitch toward it.
    /// Yaw is x-axis turn (±0.45 rad ≈ ±26°); pitch is y-axis nod
    /// (±0.12 rad ≈ ±7°). Combined with the per-particle drift, the
    /// face reads as a real being looking around the room rather than
    /// a flat rendering pinned to centre.
    pub fn step_gaze(&mut self, time: f32, dt: f32) {
        if time >= self.next_gaze_shift_at {
            let r1 = next_rand(&mut self.gaze_rng);
            let r2 = next_rand(&mut self.gaze_rng);
            let r3 = next_rand(&mut self.gaze_rng);

            // ±0.45 rad is about ±26°. Strong enough to read as a
            // real head turn from across the room.
            self.gaze_yaw_target = (r1 - 0.5) * 0.9;
            self.gaze_pitch_target = (r2 - 0.5) * 0.24;
            // 2.5-7s before the next shift — uneven so the eye doesn't
            // catch the rhythm.
            self.next_gaze_shift_at = time + 2.5 + r3 * 4.5;
        }
        // Eased lerp — gentle, never snappy. 1.3 ≈ ~750 ms to settle.
        let speed = 1.3;
        self.gaze_yaw += (self.gaze_yaw_target - self.gaze_yaw) * (dt * speed).min(1.0);
        self.gaze_pitch += (self.gaze_pitch_target - self.gaze_pitch) * (dt * speed).min(1.0);
    }

    /// Push the current audio loudness (`0..=1`) into the state.
    /// Renderer reads this each frame to compute per-particle reactive
    /// jitter — the difference between a still face and a *visibly
    /// alive* face that shimmers with speech.
    pub fn set_audio_level(&mut self, level: f32) {
        self.audio_level = level.clamp(0.0, 1.0);
    }

    /// Scatter — fire the character's transition. Each particle gets a
    /// new scatter destination via [`Character::transition`].
    pub fn scatter(&mut self, character: &Character, time: f32) {
        for (i, p) in self.particles.iter().enumerate() {
            let fnx = p.target[0];
            let fny = p.target[1];
            let off = (character.transition.displace)(fnx, fny, 1.0, time, p.seed);
            self.scattered[i] = [p.target[0] + off[0], p.target[1] + off[1], p.target[2] + off[2]];
        }
        self.materialized = 0.0;
    }

    /// Advance the per-particle materialization toward `target`
    /// (`1.0` = on the face, `0.0` = fully scattered).
    pub fn step(&mut self, character: &Character, target: f32, dt: f32) {
        self.materialized += (target - self.materialized) * (dt * 1.5).min(1.0);
        for i in 0..self.particles.len() {
            let speed = if target > self.pt[i] {
                character.transition.enter_speed
            } else {
                character.transition.exit_speed
            };
            let raw = (target - self.pt[i]) * (dt * speed).min(1.0);
            self.pt[i] += raw;

            // Per-particle interpolation between scattered and target.
            let p = self.pt[i].clamp(0.0, 1.0);
            let t = &self.particles[i].target;
            let s = &self.scattered[i];
            self.live[i] = [
                s[0] * (1.0 - p) + t[0] * p,
                s[1] * (1.0 - p) + t[1] * p,
                s[2] * (1.0 - p) + t[2] * p,
            ];
        }
    }
}

/// Settings the [`Renderer`] needs but a [`Character`] shouldn't carry
/// (per-tile concerns: dimensions, particle count, breathing clock).
#[derive(Clone, Copy, Debug)]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    /// Per-frame "trail" fill — higher = faster clear, lower = longer
    /// motion smear. Avatar's default range is 0.10..0.35.
    pub trail: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 640,
            height: 360,
            trail: 0.22,
        }
    }
}

/// Stateful renderer — owns the Pixmap so the trail can carry between
/// frames (the previous frame's darkened content is what makes the
/// particles read as motion).
pub struct Renderer {
    pixmap: Pixmap,
    settings: RenderSettings,
}

impl Renderer {
    pub fn new(settings: RenderSettings) -> Option<Self> {
        let mut pixmap = Pixmap::new(settings.width, settings.height)?;
        // Start opaque black so the per-frame fade has a stable
        // baseline and the PNG output isn't transparent where no
        // particles paint. `tiny_skia` uses premultiplied RGBA — `(0,
        // 0, 0, 255)` is "opaque black".
        for px in pixmap.data_mut().chunks_exact_mut(4) {
            px[3] = 255;
        }
        Some(Self { pixmap, settings })
    }

    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }

    /// Render one frame. `time` is monotonic seconds since renderer
    /// start — drives the breathing pulse + idle jitter.
    pub fn render(&mut self, character: &Character, state: &FaceState, time: f32) -> &Pixmap {
        let w = self.settings.width as f32;
        let h = self.settings.height as f32;

        // Trail / clear. Drop the previous frame toward black by the
        // trail alpha — gives motion smear when particles move.
        self.fade_to_black(self.settings.trail);

        // Map normalized particle space to pixels. Center the face;
        // `sc` is the smaller dimension × 0.38 (matches face-of-god).
        let cx = w * 0.5;
        let cy = h * 0.5;
        let sc = w.min(h) * 0.38;

        // Nebula cloud goes underneath everything — a dim radial glow
        // behind the face. Skipped when the character doesn't ask for
        // one.
        if let Some(neb) = character.render_config.nebula_cloud {
            self.paint_nebula(neb, cx, cy);
        }

        let breath =
            (time * character.contour_baseline.breath_rate * std::f32::consts::TAU).sin() * 0.5
                + 0.5;
        let breath_pulse = 1.0 + (breath - 0.5) * character.contour_baseline.breath_depth;

        // ── Particle pass ─────────────────────────────────────────
        let data = self.pixmap.data_mut();
        let pw = self.settings.width as usize;
        let ph = self.settings.height as usize;
        // Audio reactivity. Idle drift is large enough that a silent
        // face still visibly *crackles* in place; audio kicks add a
        // big extra Lissajous on top. Empirically tuned by watching
        // the avatar reference and matching the felt motion intensity.
        let level = state.audio_level;
        let react = level * 8.0;

        // Whole-face gentle sway — a slow ~0.5 Hz Lissajous of the
        // entire head, so it looks like she's alive and breathing
        // rather than rendered once and pinned. Independent of the
        // per-particle drift; this is gross body motion.
        let sway_x = (time * 0.45).sin() * 0.035 + (time * 0.31 + 1.7).cos() * 0.018;
        let sway_y = (time * 0.52 + 0.8).sin() * 0.022 + (time * 0.27).cos() * 0.012;
        let sway_z = (time * 0.38).sin() * 0.030;

        // Speech-onset flash factor — 1.0 right at onset, decays to
        // 0 over 400ms. Multiplied into per-particle brightness, so
        // the whole field briefly tints white when she fires a new
        // syllable. Reads as her reacting to her own speech.
        let onset_age = (time - state.speech_onset_at).max(0.0);
        let speech_flash = if onset_age < 0.4 {
            ((1.0 - onset_age / 0.4).powi(2)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Camera shake — pixel-space jitter applied to every screen
        // position. `shake` decays each frame; phase walks
        // deterministically so consecutive frames don't desync.
        let shake_x = (state.shake_phase * 1.7).sin() * state.shake;
        let shake_y = (state.shake_phase * 2.3 + 0.7).cos() * state.shake;

        for (i, p) in state.particles.iter().enumerate() {
            let pos = state.live[i];

            // Per-particle idle drift — three independent multi-axis
            // wobbles per particle so the face never reads as static.
            // Magnitudes ~5× the original (which was barely visible)
            // and a 3rd higher-frequency band gives crackle even when
            // the audio level is zero.
            let nt = time * 0.9 + level * 2.5;
            let idx = i as f32;
            let drift_x = (idx * 0.00097 + nt).sin()
                * (idx * 0.00071 + nt * 0.7).cos()
                * 0.13
                + (idx * 0.013 + time * 2.0).sin() * 0.06 * (0.6 + react)
                + (idx * 0.077 + time * 6.4).sin() * 0.025;
            let drift_y = (idx * 0.00127 + nt * 1.1).cos()
                * (idx * 0.00089 + nt * 0.5).sin()
                * 0.12
                + (idx * 0.017 + time * 2.3).cos() * 0.055 * (0.6 + react)
                + (idx * 0.083 + time * 5.7).cos() * 0.022;
            let drift_z = (idx * 0.00167 + nt * 0.8).sin() * 0.085
                + (idx * 0.021 + time * 1.7).sin() * 0.045 * (0.6 + react)
                + (idx * 0.093 + time * 4.9).sin() * 0.018;

            // Brow particles get an additional vertical offset based
            // on the current `brow` expression — positive lifts the
            // brow ridge, negative furrows it.
            let brow_offset_y = if p.is_brow { state.brow * 0.08 } else { 0.0 };

            let mut x = pos[0] + drift_x + sway_x;
            let mut y = pos[1] + drift_y + sway_y + brow_offset_y;
            let mut z = pos[2] + drift_z + sway_z;

            // Head turn (yaw + pitch). Standard 3D rotation around the
            // y-axis (yaw) and x-axis (pitch) before projection — so
            // the entire face turns as a unit toward the current gaze
            // target. Profile particles drift further left/right than
            // central ones, exactly as a real head would in motion.
            let yaw_sin = state.gaze_yaw.sin();
            let yaw_cos = state.gaze_yaw.cos();
            let pitch_sin = state.gaze_pitch.sin();
            let pitch_cos = state.gaze_pitch.cos();
            // Yaw: rotate (x, z) around y.
            let rx = x * yaw_cos + z * yaw_sin;
            let rz = -x * yaw_sin + z * yaw_cos;
            x = rx;
            z = rz;
            // Pitch: rotate (y, z) around x.
            let ry = y * pitch_cos - z * pitch_sin;
            let rz2 = y * pitch_sin + z * pitch_cos;
            y = ry;
            z = rz2;

            // Perspective project. Slightly squashed so particles in
            // front read bigger than ones behind.
            let depth = 1.0 / (2.5 + z);
            let mut sx = cx + x * sc * depth;
            // Canvas Y is inverted (top-left origin) — flip.
            let mut sy = cy - y * sc * depth;

            // Eye saccade — small screen-space offset for is_eye
            // particles only. Eyes dart slightly independent of the
            // whole head; gives a vastly more alive impression.
            if p.is_eye {
                sx += state.eye_saccade_x * sc * 0.18;
                sy -= state.eye_saccade_y * sc * 0.18;
            }

            // Camera shake — applied to every particle's screen
            // position. Cheap, reads as "the floor shook".
            sx += shake_x;
            sy += shake_y;

            // Size grows with materialization + breath + audio. The
            // audio component makes the field shimmer harder as the
            // agent speaks — same trick face-of-god-face.js uses.
            //
            // Eye particles get a much larger base size (so they
            // bloom into proper fire-orbs even with Oblivion's narrow
            // predator-slit eye geometry) plus a high-frequency
            // crackle pulse on top. Per-particle phase keeps the
            // flicker uneven across the eye region.
            let (eye_size_mul, eye_flicker) = if p.is_eye {
                let f = 1.0
                    + (time * 9.0 + p.seed * 14.0).sin().abs() * 0.85
                    + (time * 17.0 + p.seed * 31.0).sin().abs() * 0.4;
                (3.2, f)
            } else {
                (1.0, 1.0)
            };
            let size = (1.0 + state.pt[i] * 1.6 + level * 1.4)
                * depth
                * breath_pulse
                * eye_flicker
                * eye_size_mul;
            let half = size * 0.5;
            let x0 = (sx - half).floor() as i32;
            let y0 = (sy - half).floor() as i32;
            let x1 = (sx + half).ceil() as i32;
            let y1 = (sy + half).ceil() as i32;

            // Audio-reactive colour shift — each particle has a
            // pseudo-frequency response derived from its index so the
            // shift is uneven across the face (the "smoothFreq" trick
            // from face-of-god-face.js without needing an FFT). Hot
            // audio pulls the particle toward the character's
            // `audio_glow` colour proportional to its phase response.
            let p_phase = (idx * 0.07 + time * 3.7).sin().abs(); // 0..1
            let glow_t = level * character.render_config.audio_glow_strength * (0.4 + p_phase);
            let g_r = character.render_config.audio_glow[0];
            let g_g = character.render_config.audio_glow[1];
            let g_b = character.render_config.audio_glow[2];
            let mut cr = p.color[0] * (1.0 - glow_t) + g_r * glow_t;
            let mut cg = p.color[1] * (1.0 - glow_t) + g_g * glow_t;
            let mut cb = p.color[2] * (1.0 - glow_t) + g_b * glow_t;

            // Eye particles burn WHITE-HOT — fully toward (1,1,1) at
            // peak so they punch a hole in the surrounding red glow.
            // The burn stays near 1 most of the time with brief dips,
            // so the contrast stays high frame-to-frame; combined
            // with the per-frame size flicker the eyes read as
            // genuinely on fire.
            if p.is_eye {
                let burn = 0.8 + (time * 11.0 + p.seed * 23.0).sin().abs() * 0.2;
                cr = (cr + (1.0 - cr) * burn).clamp(0.0, 1.0);
                cg = (cg + (1.0 - cg) * burn * 0.92).clamp(0.0, 1.0);
                cb = (cb + (1.0 - cb) * burn * 0.55).clamp(0.0, 1.0);
            }

            let r = (cr.clamp(0.0, 1.0) * 255.0) as u32;
            let g = (cg.clamp(0.0, 1.0) * 255.0) as u32;
            let b = (cb.clamp(0.0, 1.0) * 255.0) as u32;
            // Alpha climbs with materialization + per-particle base +
            // an audio boost so the field reads brighter overall when
            // she speaks. Non-eye particles get a `0.7×` dim factor
            // so the eyes — which skip that dim — punch through with
            // a strong contrast: white-hot pinpoint over a deeper-
            // red silhouette, instead of two bright spots in a
            // uniform bright field.
            //
            // Eye particles also gate by the blink amount — when the
            // face blinks, their alpha drops to near zero for the
            // ~150ms of the blink. Instant "this is alive" cue.
            //
            // Mouth particles gate by `audio_level` — when she
            // speaks, the central mouth particles fade out, carving
            // a visible opening. Same trick the avatar SVG path uses
            // for its lip-syncing mouth aperture.
            let non_eye_dim = if p.is_eye { 1.0 } else { 0.68 };
            let blink_gate = if p.is_eye { 1.0 - state.blink } else { 1.0 };
            let mouth_gate = if p.is_mouth {
                // Aggressive falloff — even a modest audio level
                // visibly carves the mouth out.
                (1.0 - level * 2.2).max(0.05)
            } else {
                1.0
            };
            let a_f = p.color[3]
                * (0.5 + state.pt[i] * 0.5)
                * (1.0 + level * 0.6)
                * non_eye_dim
                * blink_gate
                * mouth_gate;
            if a_f < 0.01 {
                continue;
            }
            let a = (a_f.min(1.0) * 255.0).min(255.0) as u32;

            // Speech-onset flash — full-field bright tint that decays
            // over 400ms after each onset. Pushes every particle's
            // colour briefly toward white.
            let (r, g, b) = if speech_flash > 0.01 {
                let push = speech_flash * 0.55;
                let nr = (r as f32 + (255.0 - r as f32) * push) as u32;
                let ng = (g as f32 + (255.0 - g as f32) * push) as u32;
                let nb = (b as f32 + (255.0 - b as f32) * push) as u32;
                (nr, ng, nb)
            } else {
                (r, g, b)
            };

            for py in y0..=y1 {
                if py < 0 || py >= ph as i32 {
                    continue;
                }
                for px in x0..=x1 {
                    if px < 0 || px >= pw as i32 {
                        continue;
                    }
                    let off = ((py as usize) * pw + (px as usize)) * 4;
                    // Premultiplied additive-screen blend onto BGRA.
                    blend_add_premul(&mut data[off..off + 4], r as u8, g as u8, b as u8, a as u8);
                }
            }

            // ── Bloom halo ──
            // A second, larger, dimmer splat around each particle.
            // Approximates the Gaussian glow that makes the avatar
            // reference *pop* — particles read as glowing fire
            // instead of discrete dots — without paying for a full
            // blur post-pass. Skipped for already-dim particles
            // (a<40) so the halo cost only fires where it matters.
            if a >= 40 {
                let halo_size = (size * 2.2).max(4.0);
                let halo_half = halo_size * 0.5;
                let halo_a = (a as f32 * 0.22) as u8;
                if halo_a >= 4 {
                    let hx0 = ((sx - halo_half).floor() as i32).max(0);
                    let hy0 = ((sy - halo_half).floor() as i32).max(0);
                    let hx1 = ((sx + halo_half).ceil() as i32).min(pw as i32 - 1);
                    let hy1 = ((sy + halo_half).ceil() as i32).min(ph as i32 - 1);
                    for py in hy0..=hy1 {
                        let row_off = py as usize * pw;
                        for px in hx0..=hx1 {
                            let off = (row_off + px as usize) * 4;
                            blend_add_premul(
                                &mut data[off..off + 4],
                                r as u8,
                                g as u8,
                                b as u8,
                                halo_a,
                            );
                        }
                    }
                }
            }
        }

        // ── Contour pass ──────────────────────────────────────────
        // Strokes layered on top — outer glow under, sharper inner
        // line over. Pass `level` for the audio pulse and the gaze
        // angles so contours turn with the face.
        self.stroke_contours(
            character,
            breath,
            state.audio_level,
            state.gaze_yaw,
            state.gaze_pitch,
            cx,
            cy,
            sc,
        );

        // ── Accent particle ring (Utopia's lavender motes) ───────
        if let Some(ring) = character.render_config.accent_particles {
            self.draw_accent_particles(ring, time, cx, cy, sc);
        }

        // ── Embers (Oblivion's rising sparks) ────────────────────
        // Drawn AFTER the face so they composite on top, but BEFORE
        // the vignette so they dim with the corner falloff.
        if let Some(cfg) = character.render_config.embers {
            self.draw_embers(cfg, state, cx, cy, sc);
        }

        // ── Voronoi-mesh overlay (Oblivion red / Utopia gold) ────
        if let Some(mesh) = character.render_config.voronoi_mesh {
            self.draw_voronoi_overlay(mesh, time);
        }

        // ── Vignette pass — radial darkening at the corners. ─────
        // Always last so it dims everything (background, particles,
        // contours, overlays) uniformly. Heavy on Oblivion (0.72),
        // soft on Utopia (0.45).
        if character.render_config.vignette > 0.001 {
            self.apply_vignette(character.render_config.vignette);
        }

        &self.pixmap
    }

    /// Splat each live ember as a small bright spot, additively blended.
    /// Embers fade from `color_hot` → `color_cool` → alpha 0 over their
    /// lifetime. Position projects through the same perspective as the
    /// face particles so an ember spawned at z=0.1 reads as closer
    /// than the face, lending depth.
    fn draw_embers(&mut self, cfg: EmberConfig, state: &FaceState, cx: f32, cy: f32, sc: f32) {
        let pw = self.settings.width as usize;
        let ph = self.settings.height as usize;
        let data = self.pixmap.data_mut();
        for e in &state.embers {
            // Age 0..1 normalised against lifetime.
            let t = (e.age / cfg.lifetime).clamp(0.0, 1.0);
            // Bell curve — bright in the middle of life, fade in
            // quickly + fade out slowly. (1-t) gives linear decay; the
            // sin curve here gives a nicer ease.
            let alpha_curve = (1.0 - t).powi(2);
            if alpha_curve < 0.01 {
                continue;
            }
            // Colour lerp hot → cool.
            let cr = lerp_u8(cfg.color_hot[0], cfg.color_cool[0], t) as u32;
            let cg = lerp_u8(cfg.color_hot[1], cfg.color_cool[1], t) as u32;
            let cb = lerp_u8(cfg.color_hot[2], cfg.color_cool[2], t) as u32;
            let a = (alpha_curve * 220.0) as u32;
            // Project (perspective through z, just like face particles).
            let depth = 1.0 / (2.5 + e.pos[2]);
            let sx = cx + e.pos[0] * sc * depth;
            let sy = cy - e.pos[1] * sc * depth;
            // Size shrinks slightly with age — embers shrink as they
            // cool. Bigger base + more jitter than the original tune
            // so individual embers read as glowing motes, not pixels.
            let size = (5.0 * (1.0 - t * 0.4) + e.seed * 2.0) * depth;
            let half = size * 0.5;
            let x0 = (sx - half).floor() as i32;
            let y0 = (sy - half).floor() as i32;
            let x1 = (sx + half).ceil() as i32;
            let y1 = (sy + half).ceil() as i32;
            for py in y0..=y1 {
                if py < 0 || py >= ph as i32 {
                    continue;
                }
                for px in x0..=x1 {
                    if px < 0 || px >= pw as i32 {
                        continue;
                    }
                    let off = ((py as usize) * pw + (px as usize)) * 4;
                    let cr_p = (cr * a) / 255;
                    let cg_p = (cg * a) / 255;
                    let cb_p = (cb * a) / 255;
                    data[off] = (data[off] as u32 + cr_p).min(255) as u8;
                    data[off + 1] = (data[off + 1] as u32 + cg_p).min(255) as u8;
                    data[off + 2] = (data[off + 2] as u32 + cb_p).min(255) as u8;
                }
            }
        }
    }

    /// Multiply the pixmap by a radial dim factor — `1` at the centre,
    /// down to `1 - strength` at the far corners. Premultiplied RGBA,
    /// so we scale R/G/B but keep alpha at 255 (we render opaque).
    fn apply_vignette(&mut self, strength: f32) {
        let w = self.settings.width as f32;
        let h = self.settings.height as f32;
        let cx = w * 0.5;
        let cy = h * 0.5;
        let max_r = (cx * cx + cy * cy).sqrt();
        let s = strength.clamp(0.0, 1.0);
        let pw = self.settings.width as usize;
        let ph = self.settings.height as usize;
        let data = self.pixmap.data_mut();
        for py in 0..ph {
            let dy = py as f32 - cy;
            for px in 0..pw {
                let dx = px as f32 - cx;
                let r = (dx * dx + dy * dy).sqrt() / max_r;
                // Bias the falloff toward the edges so the centre stays
                // bright and only the far corners dim hard.
                let f = (1.0 - s * r * r).clamp(0.0, 1.0);
                let mul = (f * 255.0) as u32;
                let off = (py * pw + px) * 4;
                data[off] = ((data[off] as u32 * mul) / 255) as u8;
                data[off + 1] = ((data[off + 1] as u32 * mul) / 255) as u8;
                data[off + 2] = ((data[off + 2] as u32 * mul) / 255) as u8;
            }
        }
    }

    /// Paint a soft radial gradient behind the face. Manual loop over
    /// every pixel — cheap at 640×360 and lets us match the avatar
    /// exactly (premultiplied additive, soft falloff).
    fn paint_nebula(&mut self, neb: NebulaCloud, cx: f32, cy: f32) {
        let pw = self.settings.width as usize;
        let ph = self.settings.height as usize;
        let max_r = (cx.max(self.settings.width as f32 - cx)
            .powi(2)
            + cy.max(self.settings.height as f32 - cy).powi(2))
        .sqrt();
        let data = self.pixmap.data_mut();
        let nr = neb.color[0];
        let ng = neb.color[1];
        let nb = neb.color[2];
        for py in 0..ph {
            let dy = py as f32 - cy;
            for px in 0..pw {
                let dx = px as f32 - cx;
                let r = (dx * dx + dy * dy).sqrt() / max_r;
                // Soft falloff — squared so the centre is bright and
                // the corners fade gracefully.
                let strength = (1.0 - r).max(0.0).powi(2) * neb.intensity;
                if strength < 0.005 {
                    continue;
                }
                let off = (py * pw + px) * 4;
                let a = (strength * 255.0).min(255.0) as u32;
                let cr = (nr as u32 * a) / 255;
                let cg = (ng as u32 * a) / 255;
                let cb = (nb as u32 * a) / 255;
                data[off] = (data[off] as u32 + cr).min(255) as u8;
                data[off + 1] = (data[off + 1] as u32 + cg).min(255) as u8;
                data[off + 2] = (data[off + 2] as u32 + cb).min(255) as u8;
                // Alpha already 255 from fade_to_black.
            }
        }
    }

    /// Draw an orbiting ring of accent particles. Each particle has a
    /// distinct radius + phase derived from its index, so the ring
    /// reads as a *cloud* rather than a single tracked orbit.
    fn draw_accent_particles(
        &mut self,
        ring: AccentParticles,
        time: f32,
        cx: f32,
        cy: f32,
        sc: f32,
    ) {
        let data = self.pixmap.data_mut();
        let pw = self.settings.width as usize;
        let ph = self.settings.height as usize;
        let r_base = ring.radius * sc * 0.4;
        let a = (ring.alpha * 255.0).min(255.0) as u32;
        let r = ring.color[0];
        let g = ring.color[1];
        let b = ring.color[2];
        for i in 0..ring.count {
            // Deterministic per-particle phase + radius variance.
            let s = i as f32;
            let phase = (s * 0.6180339887).fract() * std::f32::consts::TAU;
            let speed = 0.18 + (s * 0.123).sin().abs() * 0.18;
            let rad = r_base * (0.85 + (s * 0.31).sin().abs() * 0.45);
            let ang = phase + time * speed;
            let px = cx + ang.cos() * rad;
            let py = cy + ang.sin() * rad * 0.85; // squashed vertically
            let pxi = px.floor() as i32;
            let pyi = py.floor() as i32;
            if pxi < 0 || pyi < 0 || (pxi as usize) >= pw || (pyi as usize) >= ph {
                continue;
            }
            // 2×2 splat — gentle and cheap.
            for dy in 0..2 {
                for dx in 0..2 {
                    let xi = pxi + dx;
                    let yi = pyi + dy;
                    if xi < 0 || yi < 0 || (xi as usize) >= pw || (yi as usize) >= ph {
                        continue;
                    }
                    let off = ((yi as usize) * pw + xi as usize) * 4;
                    let cr = (r as u32 * a) / 255;
                    let cg = (g as u32 * a) / 255;
                    let cb = (b as u32 * a) / 255;
                    data[off] = (data[off] as u32 + cr).min(255) as u8;
                    data[off + 1] = (data[off + 1] as u32 + cg).min(255) as u8;
                    data[off + 2] = (data[off + 2] as u32 + cb).min(255) as u8;
                }
            }
        }
    }

    /// Fade RGB toward black by `alpha` (0..1) while keeping the pixmap
    /// opaque. Cheap — a straight multiplication in the RGBA buffer.
    /// Alpha is held at 255 so PNG output stays opaque and the trail
    /// effect behaves like the JS Canvas version (a black fillRect
    /// every frame).
    fn fade_to_black(&mut self, alpha: f32) {
        let factor = (1.0 - alpha.clamp(0.0, 1.0)) * 255.0;
        let inv = factor as u32;
        let data = self.pixmap.data_mut();
        for px in data.chunks_exact_mut(4) {
            px[0] = ((px[0] as u32 * inv) / 255) as u8;
            px[1] = ((px[1] as u32 * inv) / 255) as u8;
            px[2] = ((px[2] as u32 * inv) / 255) as u8;
            px[3] = 255;
        }
    }

    /// Draw all contour polylines using tiny_skia's stroker. Two-pass:
    /// a fat outer-glow stroke first, a crisp inner line on top.
    fn stroke_contours(
        &mut self,
        character: &Character,
        breath: f32,
        audio: f32,
        gaze_yaw: f32,
        gaze_pitch: f32,
        cx: f32,
        cy: f32,
        sc: f32,
    ) {
        let cb: ContourBaseline = character.contour_baseline;
        // The breath modulation alone is barely visible (depth ~0.15).
        // Mixing in `audio_level` makes the contours throb harder when
        // she speaks — same trick as the avatar's contour pulse, which
        // multiplies stroke width by `1 + audio`.
        let breath_factor = 1.0 + (breath - 0.5) * cb.breath_depth;
        let audio_factor = 1.0 + audio * 0.8;
        let outer_w = cb.outer_width * breath_factor * audio_factor;
        let inner_w = cb.inner_width * breath_factor * audio_factor;
        // Boost alpha with audio too — the contour gets visibly hotter
        // (not just thicker) under speech.
        let alpha_boost = 1.0 + audio * 0.4;
        let outer_a = ((cb.outer_alpha * alpha_boost).min(1.0) * 255.0) as u8;
        let inner_a = ((cb.inner_alpha * alpha_boost).min(1.0) * 255.0) as u8;

        let band_glow = character.render_config.band_glow;

        for path in &character.contour_paths {
            if path.len() < 2 {
                continue;
            }
            // Build a tiny_skia path from the normalized polyline.
            // Contour points are 2D (treated as z=0); apply the same
            // yaw + pitch transform the particle pass uses so the
            // contour follows the head turn.
            let yaw_sin = gaze_yaw.sin();
            let yaw_cos = gaze_yaw.cos();
            let pitch_sin = gaze_pitch.sin();
            let pitch_cos = gaze_pitch.cos();
            let mut pb = PathBuilder::new();
            for (j, p) in path.iter().enumerate() {
                let mut x = p[0];
                let mut y = p[1];
                let mut z = 0.0_f32;
                // Yaw around y.
                let rx = x * yaw_cos + z * yaw_sin;
                let rz = -x * yaw_sin + z * yaw_cos;
                x = rx;
                z = rz;
                // Pitch around x.
                let ry = y * pitch_cos - z * pitch_sin;
                let rz2 = y * pitch_sin + z * pitch_cos;
                y = ry;
                z = rz2;
                let depth = 1.0 / (2.5 + z);
                let sx = cx + x * sc * depth;
                let sy = cy - y * sc * depth;
                if j == 0 {
                    pb.move_to(sx, sy);
                } else {
                    pb.line_to(sx, sy);
                }
            }
            let Some(built) = pb.finish() else { continue };

            let mut paint = Paint::default();
            paint.anti_alias = true;
            let mut stroke = Stroke::default();
            stroke.line_cap = tiny_skia::LineCap::Round;
            stroke.line_join = tiny_skia::LineJoin::Round;

            // Band glow — wider, very faint pass underneath. Gives
            // Utopia's wire bands their luminous halo.
            if band_glow {
                paint.set_color(Color::from_rgba8(
                    character.contour_color[0],
                    character.contour_color[1],
                    character.contour_color[2],
                    outer_a / 5,
                ));
                stroke.width = outer_w * 2.4;
                self.pixmap.stroke_path(&built, &paint, &stroke, Transform::identity(), None);
            }

            // Outer glow — softer + wider.
            paint.set_color(Color::from_rgba8(
                character.contour_color[0],
                character.contour_color[1],
                character.contour_color[2],
                outer_a / 2,
            ));
            stroke.width = outer_w;
            self.pixmap.stroke_path(&built, &paint, &stroke, Transform::identity(), None);

            // Inner crisp line.
            paint.set_color(Color::from_rgba8(
                character.contour_color[0],
                character.contour_color[1],
                character.contour_color[2],
                inner_a,
            ));
            stroke.width = inner_w;
            self.pixmap.stroke_path(&built, &paint, &stroke, Transform::identity(), None);
        }
    }

    /// Cracked-glass wireframe overlay. A deterministic Poisson-disk
    /// scatter of seed points (gives an organic, non-grid feel), then
    /// edges between any two seeds within a threshold distance — the
    /// proximity graph reads visually identically to a true Voronoi
    /// triangulation at this density. Each seed wobbles around its
    /// rest position over time so the mesh subtly *breathes* like a
    /// holographic lattice rather than sitting flat.
    ///
    /// Skipping the true Lloyd-relaxed Voronoi cells because at ~80
    /// seeds the proximity graph is indistinguishable to the eye but
    /// is O(N²) instead of needing a full geometric kernel. If we
    /// ever raise the density past ~200 seeds we'll need the real
    /// thing.
    fn draw_voronoi_overlay(&mut self, mesh: crate::character::VoronoiMesh, time: f32) {
        let w = self.settings.width as f32;
        let h = self.settings.height as f32;
        let mut paint = Paint::default();
        paint.set_color(Color::from_rgba8(
            mesh.color[0],
            mesh.color[1],
            mesh.color[2],
            (mesh.alpha * 255.0) as u8,
        ));
        paint.anti_alias = true;
        let mut stroke = Stroke::default();
        stroke.width = mesh.line_width;

        // ── Deterministic Poisson-disk-ish seed scatter ──
        // We use the golden-ratio low-discrepancy sequence + a quick
        // rejection step to keep seeds at least `MIN_SEP` apart. Same
        // input → same lattice, so the mesh is stable across frames.
        const TARGET_SEEDS: usize = 90;
        const PHI: f32 = 1.618_034;
        const MIN_SEP: f32 = 44.0;
        let mut seeds: Vec<[f32; 2]> = Vec::with_capacity(TARGET_SEEDS);
        let mut tried = 0;
        // The bg lattice runs slightly larger than the canvas so the
        // mesh fades off the edges rather than ending abruptly.
        let pad = MIN_SEP;
        while seeds.len() < TARGET_SEEDS && tried < TARGET_SEEDS * 8 {
            let i = tried as f32;
            let u = (i * PHI).fract();
            let v = ((i * PHI).fract() * PHI).fract();
            let x = -pad + u * (w + pad * 2.0);
            let y = -pad + v * (h + pad * 2.0);
            let ok = seeds.iter().all(|&[sx, sy]| {
                let dx = sx - x;
                let dy = sy - y;
                dx * dx + dy * dy >= MIN_SEP * MIN_SEP
            });
            if ok {
                seeds.push([x, y]);
            }
            tried += 1;
        }

        // ── Time-modulated wobble ──
        // Each seed drifts on a per-seed Lissajous so the lattice
        // shivers without losing structure.
        let live: Vec<[f32; 2]> = seeds
            .iter()
            .enumerate()
            .map(|(i, [x, y])| {
                let phase = i as f32 * 0.9;
                let amp = MIN_SEP * 0.12;
                let dx = (time * 0.4 + phase).sin() * amp;
                let dy = (time * 0.55 + phase * 1.3).cos() * amp * 0.7;
                [x + dx, y + dy]
            })
            .collect();

        // ── Proximity-graph edges ──
        // For each pair within `EDGE_MAX`, stroke a line. The cutoff
        // is chosen so each seed connects to ~5-6 neighbours on
        // average — same connectivity as a typical Voronoi cell.
        const EDGE_MAX: f32 = 78.0;
        let edge_max_sq = EDGE_MAX * EDGE_MAX;
        let mut pb = PathBuilder::new();
        for i in 0..live.len() {
            let [x1, y1] = live[i];
            for &[x2, y2] in &live[i + 1..] {
                let dx = x2 - x1;
                let dy = y2 - y1;
                let d2 = dx * dx + dy * dy;
                if d2 < edge_max_sq {
                    pb.move_to(x1, y1);
                    pb.line_to(x2, y2);
                }
            }
        }
        if let Some(path) = pb.finish() {
            self.pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }
}

/// Premultiplied alpha additive blend onto a BGRA pixel. (`tiny_skia`
/// stores in `RGBA8` byte order but premultiplied — we apply a simple
/// add for the particle pass since the field is over a dark trail.)
#[inline]
fn blend_add_premul(dst: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    let a_f = a as u32;
    // Scaled contribution.
    let cr = (r as u32 * a_f) / 255;
    let cg = (g as u32 * a_f) / 255;
    let cb = (b as u32 * a_f) / 255;
    dst[0] = (dst[0] as u32 + cr).min(255) as u8;
    dst[1] = (dst[1] as u32 + cg).min(255) as u8;
    dst[2] = (dst[2] as u32 + cb).min(255) as u8;
    dst[3] = (dst[3] as u32 + a_f).min(255) as u8;
}
