//! `ghostly` CLI — render a single frame or an animated PNG sequence
//! for any character in the registry.
//!
//! Examples:
//!   ghostly oblivion --output out.png
//!   ghostly oblivion --frames 90 --output anim/  # writes anim/0000.png … anim/0089.png
//!   ghostly --list
//!   ghostly voice oblivion --input speech.wav --output processed.wav

use std::path::PathBuf;
use std::process::ExitCode;

use ghostly::{characters, apply_emotion, CharacterPack, Emotion, FaceState, RenderSettings, Renderer};
use ghostly::audio::{profile, VoiceChain};

const USAGE: &str = "Usage: ghostly <character> [--output PATH] [--frames N] [--fps N] [--particles N] [--size WxH] [--emotion NAME[:0.0-1.0]]\n       ghostly --pack PATH.json [--output PATH] [--frames N] ...   # render a custom character pack\n       ghostly voice <character|emotion> --input PATH --output PATH\n       ghostly voice --pack PATH.json --input PATH --output PATH\n       ghostly export <character> [--output PATH.json]             # dump a built-in as an editable pack\n       ghostly --list\n\nCharacters: oblivion, narrator, utopia, eliza\nEmotions:   joy, triumph, curiosity, passion, calm, awe, warmth, concern";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    if args.iter().any(|a| a == "--list") {
        for name in characters::ALL {
            println!("{name}");
        }
        return ExitCode::SUCCESS;
    }

    // ── `voice` subcommand — offline WAV processing for testing the
    //    audio chain against a known input. The live-stream API is
    //    `ghostly::audio::VoiceChain` and takes f32 slices directly.
    if args[0] == "voice" {
        return run_voice(&args[1..]);
    }

    // ── `export` subcommand — dump a built-in character as an editable
    //    JSON pack (the starting point for forking your own).
    if args[0] == "export" {
        return run_export(&args[1..]);
    }

    // First positional arg is the character name; remaining are flags.
    let mut name: Option<String> = None;
    let mut pack_path: Option<PathBuf> = None;
    let mut output: PathBuf = PathBuf::from("out.png");
    let mut frames: usize = 1;
    let mut fps: u32 = 30;
    let mut particles: usize = 12_000;
    let mut width: u32 = 640;
    let mut height: u32 = 360;
    let mut emotion: Option<(Emotion, f32)> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with("--") && name.is_none() {
            name = Some(arg.clone());
            i += 1;
            continue;
        }
        match arg.as_str() {
            "--output" => {
                output = PathBuf::from(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--pack" => {
                pack_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--frames" => {
                frames = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(frames);
                i += 2;
            }
            "--fps" => {
                fps = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(fps);
                i += 2;
            }
            "--particles" => {
                particles = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(particles);
                i += 2;
            }
            "--size" => {
                if let Some(s) = args.get(i + 1) {
                    if let Some((w, h)) = s.split_once('x') {
                        if let (Ok(w), Ok(h)) = (w.parse(), h.parse()) {
                            width = w;
                            height = h;
                        }
                    }
                }
                i += 2;
            }
            "--emotion" => {
                // `joy` or `passion:0.75` — defaults to full intensity.
                if let Some(s) = args.get(i + 1) {
                    let (name, intensity) = match s.split_once(':') {
                        Some((n, t)) => (n, t.parse().unwrap_or(1.0_f32)),
                        None => (s.as_str(), 1.0_f32),
                    };
                    if let Some(e) = Emotion::parse(name) {
                        emotion = Some((e, intensity.clamp(0.0, 1.0)));
                    } else {
                        eprintln!("unknown emotion: {name:?}");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            _ => {
                eprintln!("unknown arg: {arg}\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }

    // Resolve the character from either a custom pack (`--pack`) or a
    // built-in name. A pack supplies its own name, so the positional
    // argument is optional when `--pack` is given.
    let character = if let Some(path) = &pack_path {
        let pack = match CharacterPack::from_file(path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("could not load pack {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        };
        match pack.to_character() {
            Some(c) => c,
            None => {
                eprintln!("pack {:?} has unknown base archetype {:?}", path.display(), pack.base);
                return ExitCode::from(2);
            }
        }
    } else {
        let Some(name) = name else {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        };
        let Some(character) = characters::by_name(&name) else {
            eprintln!("unknown character: {name:?}\n{USAGE}");
            return ExitCode::from(2);
        };
        character
    };
    let character = match emotion {
        Some((e, i)) => apply_emotion(&character, e, i),
        None => character,
    };

    let state = FaceState::new(&character, particles, 2.8, 42);
    let settings = RenderSettings {
        width,
        height,
        ..RenderSettings::default()
    };
    let Some(mut renderer) = Renderer::new(settings) else {
        eprintln!("failed to allocate {width}x{height} pixmap");
        return ExitCode::FAILURE;
    };

    if frames == 1 {
        let pixmap = renderer.render(&character, &state, 0.0);
        if let Err(e) = pixmap.save_png(&output) {
            eprintln!("save_png failed: {e}");
            return ExitCode::FAILURE;
        }
        println!("wrote {}", output.display());
    } else {
        // PNG sequence — ./out/0000.png style.
        if let Err(e) = std::fs::create_dir_all(&output) {
            eprintln!("could not create {}: {e}", output.display());
            return ExitCode::FAILURE;
        }
        // First half scattered → materialized, then steady idle.
        let mut state = state;
        let mut renderer = renderer;
        state.scatter(&character, 0.0);
        let dt = 1.0 / fps as f32;
        for f in 0..frames {
            let t = f as f32 * dt;
            // Materialize over the first second, then hold.
            let target = if t < 1.0 { t } else { 1.0 };
            state.step(&character, target, dt);
            state.step_gaze(t, dt);
            state.step_blink(t, dt);
            state.step_eye_saccade(t, dt);
            state.step_audio_onset(t, dt);
            if let Some(cfg) = character.render_config.embers {
                state.step_embers(&cfg, dt, 6.5);
            }
            let pixmap = renderer.render(&character, &state, t);
            let path = output.join(format!("{f:04}.png"));
            if let Err(e) = pixmap.save_png(&path) {
                eprintln!("save_png {} failed: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
        println!("wrote {frames} frames to {}", output.display());
    }

    ExitCode::SUCCESS
}

/// `ghostly voice <name> --input X.wav --output Y.wav` — apply the
/// character's (or named emotion's) voice chain to a WAV file. Used
/// for offline auditioning; the live-stream path uses
/// [`ghostly::audio::VoiceChain`] directly.
fn run_voice(args: &[String]) -> ExitCode {
    let usage = "Usage: ghostly voice <character|emotion> --input PATH --output PATH\n       ghostly voice --pack PATH.json --input PATH --output PATH";
    if args.is_empty() {
        eprintln!("{usage}");
        return ExitCode::from(2);
    }
    let mut name: Option<String> = None;
    let mut pack_path: Option<PathBuf> = None;
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with("--") && name.is_none() {
            name = Some(arg.clone());
            i += 1;
            continue;
        }
        match arg.as_str() {
            "--pack" => {
                pack_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--input" => {
                input = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--output" => {
                output = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}\n{usage}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(input) = input else {
        eprintln!("--input required\n{usage}");
        return ExitCode::from(2);
    };
    let Some(output) = output else {
        eprintln!("--output required\n{usage}");
        return ExitCode::from(2);
    };

    // Resolve the voice profile from a custom pack (`--pack`) or a name.
    // For a name: try character-default first, then direct emotion
    // lookup — so `ghostly voice oblivion` and `ghostly voice passion`
    // both produce passion-chained audio.
    let prof = if let Some(path) = &pack_path {
        match CharacterPack::from_file(path) {
            Ok(p) => p.voice_profile(),
            Err(e) => {
                eprintln!("could not load pack {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
    } else {
        let Some(name) = name.as_deref() else {
            eprintln!("character/emotion name or --pack required\n{usage}");
            return ExitCode::from(2);
        };
        match name {
            "narrator" | "utopia" | "oblivion" | "eliza" => profile::for_character(name),
            _ => profile::for_emotion(name),
        }
    };

    // Read WAV — convert to mono f32 regardless of source format.
    let mut reader = match hound::WavReader::open(&input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("could not open {}: {e}", input.display());
            return ExitCode::FAILURE;
        }
    };
    let spec = reader.spec();
    let sr = spec.sample_rate as f32;
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };
    // Downmix to mono.
    let mut mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // Process — flush a short tail so reverb/delay/shimmer can decay.
    let tail_s = 1.5;
    let tail_samples = (tail_s * sr) as usize;
    mono.extend(std::iter::repeat(0.0).take(tail_samples));
    let mut chain = VoiceChain::new(prof, sr);
    chain.process(&mut mono);

    // Soft-clip the output (compressor-out should keep us in range
    // but tanh provides a safety net against any chain overshoot).
    for s in mono.iter_mut() {
        *s = s.tanh();
    }

    // Write 16-bit mono WAV — matches what every player will pick up.
    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: spec.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = match hound::WavWriter::create(&output, out_spec) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("could not create {}: {e}", output.display());
            return ExitCode::FAILURE;
        }
    };
    for s in mono.iter() {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        if writer.write_sample(v).is_err() {
            eprintln!("write_sample failed");
            return ExitCode::FAILURE;
        }
    }
    if let Err(e) = writer.finalize() {
        eprintln!("finalize failed: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "wrote {} ({} samples, {} Hz, profile={})",
        output.display(),
        mono.len(),
        spec.sample_rate,
        prof.label
    );
    ExitCode::SUCCESS
}

/// `ghostly export <character> [--output PATH.json]` — dump a built-in
/// character as an editable JSON pack. This is the starting point for
/// forking: export, tweak geometry/palette/effects/voice, then
/// `ghostly render --pack PATH.json`. Writes to stdout if `--output`
/// is omitted.
fn run_export(args: &[String]) -> ExitCode {
    let usage = "Usage: ghostly export <character> [--output PATH.json]";
    let Some(name) = args.first() else {
        eprintln!("{usage}\nCharacters: {}", characters::ALL.join(", "));
        return ExitCode::from(2);
    };
    let mut output: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                output = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}\n{usage}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(pack) = CharacterPack::from_character(name) else {
        eprintln!("unknown character: {name:?}\nCharacters: {}", characters::ALL.join(", "));
        return ExitCode::from(2);
    };
    let json = match pack.to_json_string() {
        Ok(j) => j,
        Err(e) => {
            eprintln!("could not serialize pack: {e}");
            return ExitCode::FAILURE;
        }
    };
    match output {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, json) {
                eprintln!("could not write {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
            println!("wrote {}", path.display());
        }
        None => println!("{json}"),
    }
    ExitCode::SUCCESS
}
